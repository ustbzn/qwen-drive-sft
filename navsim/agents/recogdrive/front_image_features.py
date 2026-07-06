"""Front-camera image feature builder for ReCogDrive (experimental).

把当前帧前视图（默认 cam_f0）resize 成固定分辨率的 uint8 张量缓存下来，供规划侧的
可训练 ResNet 图像编码器（recogdrive_diffusion_planner.ImageEncoder）编成 context token，
与 VLM / obj / hdmap token 并列进 DiT cross-attention。

设计原则（对齐 object_vector_features.ObjectVectorFeatureBuilder 的范式）：
- 只做数据：读图 + resize，不跑任何网络；ResNet 在模型侧、端到端训练。
- 缓存 uint8 (3,H,W)（不缓存 float32，磁盘省 4 倍），在 ImageEncoder.forward 内再
  转 float + ImageNet 归一化。
- compute_features(agent_input) / compute_features_from_scene(scene) 均可取图，缺图时返回
  全零占位 + front_image_valid=False，保证无图样本不崩（对应 token 会被 mask 掉）。

关键前提：当前 IL/RL 走 CacheOnlyDataset，缓存里没有像素。启用 use_image_encoder 需在
全链路（缓存 / IL / RL / eval）一致开启，并重跑 hidden-state 缓存补出本 builder 的 .gz
（VLM 不重算）。CacheOnlyDataset 缺该 .gz 会静默丢样本。
"""

from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


# --- collate / forward 共用的键名，单一来源，避免各处拼写漂移 ---
FRONT_IMAGE_KEYS = ("front_image", "front_image_valid")


class FrontImageFeatureBuilder(AbstractFeatureBuilder):
    """读当前帧前视图，输出固定形状 (3,H,W) uint8 图像张量 + 有效位。"""

    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 704,
        view: str = "cam_f0",
    ):
        """
        :param image_height: resize 目标高（H）
        :param image_width: resize 目标宽（W）
        :param view: 取图相机字段名，默认前视 cam_f0
        """
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.view = str(view)

    def get_unique_name(self) -> str:
        # 缓存文件名带分辨率 / 视角，改分辨率或视角即换目录，避免脏读旧缓存。
        return f"front_image_feature_v1_{self.view}_{self.image_height}x{self.image_width}"

    # ------------------------------------------------------------------ #
    # 离线缓存 / 在线 eval（有 scene）路径：复用 agent_input 取图逻辑
    # ------------------------------------------------------------------ #
    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        return self.compute_features(scene.get_agent_input())

    # ------------------------------------------------------------------ #
    # 从 agent_input 当前帧取前视图
    # ------------------------------------------------------------------ #
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        arr = None
        cameras = getattr(agent_input, "cameras", None)
        if cameras:
            cam_bundle = cameras[-1]
            camera_obj = getattr(cam_bundle, self.view, None)
            image_field = getattr(camera_obj, "image", None) if camera_obj is not None else None
            arr = self._load_image(image_field)

        if arr is None:
            return self._empty_feature_dict()

        # (H,W,3) uint8 -> (3,H,W) uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return {
            "front_image": tensor,
            "front_image_valid": torch.tensor(True, dtype=torch.bool),
        }

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _empty_feature_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "front_image": torch.zeros((3, self.image_height, self.image_width), dtype=torch.uint8),
            "front_image_valid": torch.tensor(False, dtype=torch.bool),
        }

    def _load_image(self, image_field) -> Optional[np.ndarray]:
        """把 Camera.image（路径 str/Path 或已加载的 ndarray）读成 (H,W,3) uint8。失败返回 None。"""
        try:
            if image_field is None:
                return None
            if isinstance(image_field, np.ndarray):
                img = Image.fromarray(image_field)
            else:
                s = str(image_field)
                if not s or s == "None":
                    return None
                img = Image.open(s)
            # PIL.resize 参数是 (W, H)
            img = img.convert("RGB").resize((self.image_width, self.image_height))
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            return None


def collate_front_image_features(features: Dict[str, torch.Tensor], features_list) -> None:
    """把固定形状的前视图张量 stack 进 batch。仅当缓存里带 front_image 时补（零影响保护）。

    front_image 是 (3,H,W) uint8 固定形状，直接 torch.stack；front_image_valid 是标量 bool。
    与 hdmap 变长特征不同，无需 pad_sequence。IL / RL / accelerate / ema 四处 collate 共用本函数。
    """
    if not features_list:
        return
    if "front_image" in features_list[0]:
        features["front_image"] = torch.stack(
            [f["front_image"] for f in features_list], dim=0
        ).cpu()
        features["front_image_valid"] = torch.stack(
            [f["front_image_valid"] for f in features_list], dim=0
        ).cpu()
