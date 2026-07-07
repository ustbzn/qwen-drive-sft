from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.common.command_utils import command_code_to_text
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder, select_hidden_state
from .hdmap_vector_features import (
    HDMAP_VECTOR_BOOL_KEYS,
    HDMAP_VECTOR_FLOAT_KEYS,
    HDMAP_VECTOR_LONG_KEYS,
    HDMapVectorFeatureBuilder,
)
from .object_vector_features import (
    ObjectVectorFeatureBuilder,
    OBJECT_FEATURE_DIM,
    OBJECT_NUM_TYPES,
)
from .object_temporal_features import ObjectTemporalFeatureBuilder
from .front_image_features import FrontImageFeatureBuilder
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_view_utils import (
    collect_view_frame_paths,
    flatten_view_frames,
    view_labels_for,
    build_history_str,
    build_driving_question,
)
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single',
        vlm_type: Optional[str] = 'internvl',
        dit_type: Optional[str] = 'small',
        sampling_method: Optional[str] = 'ddim',
        cache_mode: bool = False,
        cache_hidden_state: bool = True,
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '',
        reference_policy_checkpoint: Optional[str] = '',
        vlm_size: Optional[str] = 'small',
        train_backbone: bool = False,
        vl_hidden_dim: Optional[int] = None,
        normalize_vl_hidden_states: bool = False,
        vlm_ablation: str = "none",
        use_vlm: bool = True,
        first_hidden_state_index: int = 0,
        last_hidden_state_index: int = -1,
        bc_coeff: float = 0.1,
        bc_teacher_deterministic: bool = False,
        view_list: Optional[List[str]] = None,
        num_frames: int = 1,
        use_video_input: bool = False,
        video_fps: float = 2.0,
        use_hdmap_vector: bool = False,
        max_map_tokens: int = 128,
        map_sample_points: int = 20,
        hdmap_radius: float = 120.0,
        strict_hdmap_vector: bool = False,
        use_lane_boundaries: bool = False,
        use_object_vector: bool = False,
        max_obj_tokens: int = 64,
        obj_range_x: Optional[List[float]] = None,
        obj_range_y: Optional[List[float]] = None,
        include_static_objects: bool = False,
        object_token_dropout: float = 0.1,
        object_encoder_type: str = "mlp",
        static_speed_threshold: float = 0.5,
        ego_status_encoder_type: str = 'mlp',
        ego_state_dropout: float = 0.75,
        ego_num_command_states: int = 4,
        use_image_encoder: bool = False,
        image_encoder_arch: str = "resnet34",
        image_encoder_pretrained: bool = True,
        image_height: int = 256,
        image_width: int = 704,
        image_grid_h: int = 8,
        image_grid_w: int = 16,
        image_token_dropout: float = 0.0,
        image_view: str = "cam_f0",
    ):
        super().__init__(requires_scene=bool(use_object_vector))
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode    # 是否开启缓存机制
        self.cache_hidden_state = cache_hidden_state    # 是否缓存 backbone hidden states
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone
        self.vl_hidden_dim = vl_hidden_dim
        self.normalize_vl_hidden_states = normalize_vl_hidden_states
        self.vlm_ablation = vlm_ablation
        # VLM 总开关（默认 True）。False：完全不加载 / 不前向 VLM（不构造在线 backbone），
        # planner 也不构造 VLM 投影层，DiT context 仅由 obj/hdmap/img token 组成。与 vlm_ablation
        # 正交：use_vlm=False 直接不用 VLM（更小更省），vlm_ablation=zero 仍加载并投影后置零。
        self.use_vlm = bool(use_vlm)
        self.first_hidden_state_index = int(first_hidden_state_index)
        self.last_hidden_state_index = int(last_hidden_state_index)
        self.bc_coeff = bc_coeff
        self.bc_teacher_deterministic = bc_teacher_deterministic
        # 多视角 / 多帧配置（默认单视角单帧，零影响）。视角数 = len(view_list)。
        self.view_list = list(view_list) if view_list else ["cam_f0"]
        self.num_frames = int(num_frames)
        self.use_video_input = bool(use_video_input)
        self.video_fps = float(video_fps)
        self.use_hdmap_vector = bool(use_hdmap_vector)
        self.max_map_tokens = int(max_map_tokens)
        self.map_sample_points = int(map_sample_points)
        self.hdmap_radius = float(hdmap_radius)
        self.strict_hdmap_vector = bool(strict_hdmap_vector)
        self.use_lane_boundaries = bool(use_lane_boundaries)
        # 障碍物 obj token 配置（默认关闭，零影响）。开启则 requires_scene=True，
        # eval / 缓存时从 Scene 当前帧 annotations 取障碍物。
        self.use_object_vector = bool(use_object_vector)
        self.max_obj_tokens = int(max_obj_tokens)
        self.obj_range_x = tuple(obj_range_x) if obj_range_x is not None else (-20.0, 100.0)
        self.obj_range_y = tuple(obj_range_y) if obj_range_y is not None else (-40.0, 40.0)
        self.include_static_objects = bool(include_static_objects)
        self.object_token_dropout = float(object_token_dropout)
        # obj 编码器类型: "mlp"=单帧几何(ObjectEncoder); "temporal"=差分+时序FPN+静态MLP(TemporalObjectEncoder)
        self.object_encoder_type = str(object_encoder_type)
        self.static_speed_threshold = float(static_speed_threshold)
        # 自车状态编码器(默认 mlp，零影响)。attention=SDE 状态丢弃编码器：训练时随机遮蔽
        # 运动学量、导航命令永久可见，抑制"外推当前自车运动学"的捷径；与 vlm_ablation 正交
        # (none/zero 均生效)。切换类型会改变 ego_status_encoder 权重结构，须重训 IL 并让
        # RL/eval 用同一类型(否则 attention 层被 initialize() 的 strict=False 丢成随机初始化)。
        self.ego_status_encoder_type = str(ego_status_encoder_type)
        self.ego_state_dropout = float(ego_state_dropout)
        self.ego_num_command_states = int(ego_num_command_states)
        # 前视图像 ResNet encoder（默认关闭，零影响；实验用）。图像从 agent_input.cameras
        # 当前帧取，不依赖 Scene，故不改 requires_scene。开启需全链路（缓存/IL/RL/eval）一致，
        # 并重跑 hidden-state 缓存补出 front_image_feature_*.gz（VLM 不重算）。
        self.use_image_encoder = bool(use_image_encoder)
        self.image_encoder_arch = str(image_encoder_arch)
        self.image_encoder_pretrained = bool(image_encoder_pretrained)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.image_grid_h = int(image_grid_h)
        self.image_grid_w = int(image_grid_w)
        self.image_token_dropout = float(image_token_dropout)
        self.image_view = str(image_view)

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode and self.use_vlm:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )

            if not self.train_backbone:     # 如果不需要训练 backbone，则冻结 backbone 参数，只训练 diffusion planner
                for p in self.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True
        elif not self.cache_hidden_state and not self.cache_mode and not self.use_vlm:
            # use_vlm=False 的在线 eval / 可视化：不加载 Qwen3，backbone 保持 None，
            # forward 走无 VLM 分支（hidden state 置 None）。
            print("Agent running in 'no-cache' mode with use_vlm=False. Skipping VLM backbone.")

        # action 为 [x, y, heading]，因此action_dim=3; action_horizon=8 表示预测未来8个时间步的轨迹，每一个时间步为0.5s;
        if self.vl_hidden_dim is None:
            if self.vlm_type == "qwen3":
                self.vl_hidden_dim = 2048
            elif self.vlm_size == "large":
                self.vl_hidden_dim = 3584
            else:
                self.vl_hidden_dim = 1536

        if self.dit_type == "large":
            cfg = make_recogdrive_config(
                self.dit_type,
                action_dim=3,
                action_horizon=8,
                grpo=self.grpo,
                input_embedding_dim=1536,
                sampling_method=sampling_method,
                vl_hidden_dim=self.vl_hidden_dim,
                normalize_vl_hidden_states=self.normalize_vl_hidden_states,
                use_hdmap_vector=self.use_hdmap_vector,
                max_map_tokens=self.max_map_tokens,
                map_sample_points=self.map_sample_points,
                use_lane_boundaries=self.use_lane_boundaries,
                use_object_vector=self.use_object_vector,
                obj_feature_dim=OBJECT_FEATURE_DIM,
                obj_num_types=OBJECT_NUM_TYPES,
                object_token_dropout=self.object_token_dropout,
                object_encoder_type=self.object_encoder_type,
                static_speed_threshold=self.static_speed_threshold,
                ego_status_encoder_type=self.ego_status_encoder_type,
                ego_state_dropout=self.ego_state_dropout,
                ego_num_command_states=self.ego_num_command_states,
                use_image_encoder=self.use_image_encoder,
                image_encoder_arch=self.image_encoder_arch,
                image_encoder_pretrained=self.image_encoder_pretrained,
                image_grid_h=self.image_grid_h,
                image_grid_w=self.image_grid_w,
                image_token_dropout=self.image_token_dropout,
            )
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(
                self.dit_type,
                action_dim=3,
                action_horizon=8,
                grpo=self.grpo,
                input_embedding_dim=384,
                sampling_method=sampling_method,
                vl_hidden_dim=self.vl_hidden_dim,
                normalize_vl_hidden_states=self.normalize_vl_hidden_states,
                use_hdmap_vector=self.use_hdmap_vector,
                max_map_tokens=self.max_map_tokens,
                map_sample_points=self.map_sample_points,
                use_lane_boundaries=self.use_lane_boundaries,
                use_object_vector=self.use_object_vector,
                obj_feature_dim=OBJECT_FEATURE_DIM,
                obj_num_types=OBJECT_NUM_TYPES,
                object_token_dropout=self.object_token_dropout,
                object_encoder_type=self.object_encoder_type,
                static_speed_threshold=self.static_speed_threshold,
                ego_status_encoder_type=self.ego_status_encoder_type,
                ego_state_dropout=self.ego_state_dropout,
                ego_num_command_states=self.ego_num_command_states,
                use_image_encoder=self.use_image_encoder,
                image_encoder_arch=self.image_encoder_arch,
                image_encoder_pretrained=self.image_encoder_pretrained,
                image_grid_h=self.image_grid_h,
                image_grid_w=self.image_grid_w,
                image_token_dropout=self.image_token_dropout,
            )

        cfg.vlm_size = self.vlm_size
        cfg.vlm_ablation = self.vlm_ablation
        cfg.use_vlm = self.use_vlm

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            cfg.grpo_cfg.bc_coeff = self.bc_coeff
            cfg.grpo_cfg.bc_teacher_deterministic = self.bc_teacher_deterministic

        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()      # 将整个Diffusion Planner存成agent的action_head，供后续forward等函数使用
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:    # 获取传感器配置，声明这个 agent 需要哪些传感器。
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]  # 声明训练时要算哪些"目标"，这里只做轨迹预测

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:     # 声明怎么从原始数据变成模型输入
        builders: List[AbstractFeatureBuilder] = [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
            first_hidden_state_index=self.first_hidden_state_index,
            last_hidden_state_index=self.last_hidden_state_index,
            view_list=self.view_list,
            num_frames=self.num_frames,
            use_video_input=self.use_video_input,
            video_fps=self.video_fps,
        )]
        if self.use_hdmap_vector:
            builders.append(
                HDMapVectorFeatureBuilder(
                    max_map_tokens=self.max_map_tokens,
                    map_sample_points=self.map_sample_points,
                    radius=self.hdmap_radius,
                    strict=self.strict_hdmap_vector,
                )
            )

        if self.use_object_vector:
            if self.object_encoder_type == "temporal":
                builders.append(ObjectTemporalFeatureBuilder(
                    max_obj_tokens=self.max_obj_tokens,
                    obj_range_x=self.obj_range_x,
                    obj_range_y=self.obj_range_y,
                    include_static=self.include_static_objects,
                    static_speed_threshold=self.static_speed_threshold,
                ))
            else:
                builders.append(ObjectVectorFeatureBuilder(
                    max_obj_tokens=self.max_obj_tokens,
                    obj_range_x=self.obj_range_x,
                    obj_range_y=self.obj_range_y,
                    include_static=self.include_static_objects,
                ))

        if self.use_image_encoder:
            builders.append(FrontImageFeatureBuilder(
                image_height=self.image_height,
                image_width=self.image_width,
                view=self.image_view,
            ))
        return builders

    def _build_action_inputs(
        self,
        features: Dict[str, torch.Tensor],
        status_feature: torch.Tensor,
        model_dtype: torch.dtype,
        action: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        action_data: Dict[str, torch.Tensor] = {"status_feature": status_feature.to(model_dtype)}
        if action is not None:
            action_data["action"] = action.to(model_dtype)
        if self.use_hdmap_vector:
            for key in HDMAP_VECTOR_FLOAT_KEYS:
                if key in features:
                    action_data[key] = features[key].to(model_dtype)
            for key in HDMAP_VECTOR_LONG_KEYS:
                if key in features:
                    action_data[key] = features[key].long()
            for key in HDMAP_VECTOR_BOOL_KEYS:
                if key in features:
                    action_data[key] = features[key].bool()
        return BatchFeature(data=action_data)
    # 从 features 拿到或算出 VLM 的 first/last hidden states，拼好 action inputs，交给 Diffusion Planner 做轨迹预测。
    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        # 优先用已算好的 hidden_state：缓存模式从缓存读；在线 eval 多视角 / 多帧时
        # 由 compute_trajectory 调 _encode_views 预先算好并注入 features（方案 B）。
        # use_vlm=False：完全不使用 VLM hidden state（不读缓存、不在线跑 backbone），置 None
        # 交给 planner 走无 VLM 分支。训练时缓存里仍带 hidden state，这里直接忽略即可。
        if not self.use_vlm:
            first_hidden_state = None
            last_hidden_state = None
        elif self.cache_hidden_state or ("first_hidden_state" in features):
            first_hidden_state = features["first_hidden_state"].cuda()
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:   # 现场跑backbone
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor) # 图片路径解码 tensor → list[str]
            
            command_str_list = [command_code_to_text(command) for command in high_command_one_hot]

            # 为 batch 内每个样本拼一个 prompt（questions）
            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            outputs = self.backbone.forward_from_paths(image_paths, questions)
            first_hidden_state = select_hidden_state(outputs.hidden_states, self.first_hidden_state_index)
            last_hidden_state = select_hidden_state(outputs.hidden_states, self.last_hidden_state_index)
        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if self.use_vlm:
            if first_hidden_state.ndim == 2:
                first_hidden_state = first_hidden_state.unsqueeze(0)
            if last_hidden_state.ndim == 2:
                last_hidden_state = last_hidden_state.unsqueeze(0)

            first_hidden_state = first_hidden_state.to(model_dtype)
            last_hidden_state = last_hidden_state.to(model_dtype)

        # 障碍物 obj 字段（启用 obj token 且 features 里带 obj 时才注入）。obj_features 跟随
        # model_dtype，obj_type/obj_valid_mask 保持 long/bool。
        obj_data: Dict[str, torch.Tensor] = {}
        if self.use_object_vector and "obj_features" in features:
            obj_data = {
                "obj_features": features["obj_features"].to(model_dtype),
                "obj_type": features["obj_type"].long(),
                "obj_valid_mask": features["obj_valid_mask"].bool(),
            }
            # temporal 编码器额外注入差分序列与动静标记
            if self.object_encoder_type == "temporal" and "obj_diff_features" in features:
                obj_data["obj_diff_features"] = features["obj_diff_features"].to(model_dtype)
                obj_data["obj_is_dynamic"] = features["obj_is_dynamic"].bool()

        # 前视图像字段（启用图像 encoder 且 features 里带图时才注入）。front_image 保持 uint8
        # （ImageEncoder 内部转 float + 归一化，省显存/带宽）；front_image_valid 为 bool。
        img_data: Dict[str, torch.Tensor] = {}
        if self.use_image_encoder and "front_image" in features:
            img_data = {
                "front_image": features["front_image"],
                "front_image_valid": features["front_image_valid"].bool(),
            }

        if self.training and not self.grpo:    # Planner.forward 训练
            action_inputs = self._build_action_inputs(features, status_feature, model_dtype, targets["trajectory"])
            # 合并 obj_data 到 action_inputs
            if obj_data:
                action_inputs.data.update(obj_data)
            if img_data:
                action_inputs.data.update(img_data)
            return self.action_head(first_hidden_state, last_hidden_state, action_inputs)
        elif self.training and self.grpo: # GRPO 训练
            action_inputs = self._build_action_inputs(features, status_feature, model_dtype, targets["trajectory"])
            # 合并 obj_data 到 action_inputs
            if obj_data:
                action_inputs.data.update(obj_data)
            if img_data:
                action_inputs.data.update(img_data)
            return self.action_head.forward_grpo(first_hidden_state, last_hidden_state, action_inputs, tokens_list)    # tokens_list 是场景的token，在计算 PDM score 需要知道"这个样本对应哪个场景/片段"的 token，用 token 去加载对应的 metric cache
        else:    # 推理，无 action
            action_inputs = self._build_action_inputs(features, status_feature, model_dtype)
            # 合并 obj_data 到 action_inputs
            if obj_data:
                action_inputs.data.update(obj_data)
            if img_data:
                action_inputs.data.update(img_data)
            return self.action_head.get_action(first_hidden_state, last_hidden_state, action_inputs)

    def _is_online_backbone(self) -> bool:
        """在线 backbone 模式：eval / 可视化时每样本现跑 VLM（非缓存读、非 cache_mode 写）。"""
        return (not self.cache_hidden_state) and (not self.cache_mode)

    def _needs_view_encoding(self) -> bool:
        """是否需要走 _encode_views（多视角或多帧）。单视角单帧时保持旧 image_path_tensor
        路径以零影响。"""
        return self._is_online_backbone() and (
            len(self.view_list) > 1 or int(self.num_frames) > 1
        )

    def _encode_views(self, agent_input: AgentInput):
        """从 agent_input.cameras 取多视角 / 多帧图，现跑 backbone，返回 (first_hs, last_hs)。

        与 features 缓存分支共用 view_utils，prompt 文案一致（方案 B：在线 eval 单样本
        直接持有 agent_input，无需经 image_path_tensor）。
        """
        if self.backbone is None:
            raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")

        cameras = agent_input.cameras
        ego_statuses = agent_input.ego_statuses
        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32,
        )
        command = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        command_str = command_code_to_text(command)
        history_str = build_history_str(history_trajectory)

        view_frame_paths = collect_view_frame_paths(cameras, self.view_list, self.num_frames)
        flat = flatten_view_frames(view_frame_paths)
        question = build_driving_question(
            history_str, command_str, self.view_list, self.num_frames,
            model_type=self.vlm_type, flat_image_count=len(flat),
        )
        outputs = self.backbone.forward_from_paths(
            [view_frame_paths], [question],
            view_labels=view_labels_for(self.view_list),
            num_frames=self.num_frames,
            use_video_input=self.use_video_input,
            video_fps=self.video_fps,
        )
        first_hs = select_hidden_state(outputs.hidden_states, self.first_hidden_state_index)
        last_hs = select_hidden_state(outputs.hidden_states, self.last_hidden_state_index)
        return first_hs.squeeze(0).float().cpu(), last_hs.squeeze(0).float().cpu()

    def compute_trajectory(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features。hdmap/obj builder 在 scene 可用时走 compute_features_from_scene；
        # scene 为 None 时返回空对象，模型退化到 VLM+ego。
        for builder in self.get_feature_builders():
            if scene is not None and hasattr(builder, "compute_features_from_scene"):
                features.update(builder.compute_features_from_scene(scene))
            else:
                features.update(builder.compute_features(agent_input))

        # 多视角 / 多帧在线 eval：用 _encode_views 算 hidden_state 注入，绕开单路径 image_path_tensor
        if self._needs_view_encoding():
            first_hs, last_hs = self._encode_views(agent_input)
            features["first_hidden_state"] = first_hs
            features["last_hidden_state"] = last_hs
            features.pop("image_path_tensor", None)
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features（同 compute_trajectory 的 scene-aware 逻辑）
        for builder in self.get_feature_builders():
            if scene is not None and hasattr(builder, "compute_features_from_scene"):
                features.update(builder.compute_features_from_scene(scene))
            else:
                features.update(builder.compute_features(agent_input))

        if self._needs_view_encoding():
            first_hs, last_hs = self._encode_views(agent_input)
            features["first_hidden_state"] = first_hs
            features["last_hidden_state"] = last_hs
            features.pop("image_path_tensor", None)

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))

        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=20, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
    vl_hidden_dim: int = 1536,
    normalize_vl_hidden_states: bool = False,
    use_hdmap_vector: bool = False,
    max_map_tokens: int = 128,
    map_sample_points: int = 20,
    use_lane_boundaries: bool = False,
    use_object_vector: bool = False,
    obj_feature_dim: int = OBJECT_FEATURE_DIM,
    obj_num_types: int = OBJECT_NUM_TYPES,
    object_token_dropout: float = 0.1,
    object_encoder_type: str = "mlp",
    obj_temporal_dim: int = 128,
    static_speed_threshold: float = 0.5,
    ego_status_encoder_type: str = 'mlp',
    ego_state_dropout: float = 0.75,
    ego_num_command_states: int = 4,
    use_image_encoder: bool = False,
    image_encoder_arch: str = "resnet34",
    image_encoder_pretrained: bool = True,
    image_grid_h: int = 8,
    image_grid_w: int = 16,
    image_token_dropout: float = 0.0,
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        # 小尺寸DiT：8头注意力，每个头维度48，16层，输出维度512（最后一层输出维度）
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        # 大尺寸DiT：32头注意力，每个头维度48，16层，输出维度1536（最后一层输出维度）
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    # 在 DiT 中，对于偶数层，只做 self-attention：Q、K、V 都是轨迹 hidden_states（和自己）。
    # 对于奇数层，做 cross-attention：Q = 轨迹，K/V = VLM 的 hidden_states。
    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,   # 在 DiT 里交替做 self-attn 和 cross-attn（奇数层用 cross-attn 看 VLM 特征，偶数层不用 cross-attn）。
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
        vl_hidden_dim=vl_hidden_dim,
        normalize_vl_hidden_states=normalize_vl_hidden_states,
        use_hdmap_vector=use_hdmap_vector,
        max_map_tokens=max_map_tokens,
        map_sample_points=map_sample_points,
        use_lane_boundaries=use_lane_boundaries,
        use_object_vector=use_object_vector,
        obj_feature_dim=obj_feature_dim,
        obj_num_types=obj_num_types,
        object_token_dropout=object_token_dropout,
        object_encoder_type=object_encoder_type,
        obj_temporal_dim=obj_temporal_dim,
        static_speed_threshold=static_speed_threshold,
        ego_status_encoder_type=ego_status_encoder_type,
        ego_state_dropout=ego_state_dropout,
        ego_num_command_states=ego_num_command_states,
        use_image_encoder=use_image_encoder,
        image_encoder_arch=image_encoder_arch,
        image_encoder_pretrained=image_encoder_pretrained,
        image_grid_h=image_grid_h,
        image_grid_w=image_grid_w,
        image_token_dropout=image_token_dropout,
    )

    return config
