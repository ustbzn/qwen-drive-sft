"""Camera image feature builder for ReCogDrive (experimental).

把当前帧一个或多个视角 resize 成固定分辨率的 uint8 张量缓存下来，供规划侧的
可训练 ResNet 图像编码器（recogdrive_diffusion_planner.ImageEncoder）编成 context token，
与 VLM / obj / hdmap token 并列进 DiT cross-attention。

设计原则（对齐 object_vector_features.ObjectVectorFeatureBuilder 的范式）：
- 只做数据：读图 + resize，不跑任何网络；ResNet 在模型侧、端到端训练。
- 单视角继续缓存 uint8 (3,H,W)，保持旧 front_image_feature_v1_cam_f0_*.gz 完全兼容；
  多视角缓存 (V,3,H,W)。在 ImageEncoder.forward 内再转 float + ImageNet 归一化。
- compute_features(agent_input) / compute_features_from_scene(scene) 均可取图，缺图时返回
  全零占位 + front_image_valid=False，保证无图样本不崩（对应 token 会被 mask 掉）。

关键前提：当前 IL/RL 走 CacheOnlyDataset，缓存里没有像素。启用 use_image_encoder 需在
全链路（缓存 / IL / RL / eval）一致开启，并补跑同一特征缓存目录生成本 builder 的
.gz；已有 hidden-state 文件会按 builder 跳过，不会重算。CacheOnlyDataset 缺该 .gz 会报告缺失并过滤
不完整样本。
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


# --- collate / forward 共用的键名，单一来源，避免各处拼写漂移 ---
FRONT_IMAGE_KEYS = ("front_image", "front_image_valid")


def normalize_image_view_list(
    image_view: Optional[str] = None,
    image_view_list: Optional[Sequence[str]] = None,
) -> List[str]:
    """解析新旧图像视角配置，显式旧参数优先以保持向后兼容。"""
    if image_view:
        raw_views = [image_view]
    elif isinstance(image_view_list, str):
        # 兼容 shell 环境变量的 ``cam_f0,cam_b0`` 与 Hydra ``[cam_f0,cam_b0]``。
        text = image_view_list.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_views = [item.strip().strip("'\"") for item in text.split(",") if item.strip()]
    else:
        raw_views = list(image_view_list or ["cam_f0"])
    views = [str(item).strip() for item in raw_views]
    if not views or any(not item for item in views):
        raise ValueError("image_view_list must contain at least one non-empty camera name")
    if len(set(views)) != len(views):
        raise ValueError(f"image_view_list contains duplicate cameras: {views}")
    return views


class FrontImageFeatureBuilder(AbstractFeatureBuilder):
    """读取当前帧相机图像；单视角保持旧缓存契约，多视角增加 view 维。"""

    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 704,
        view: Optional[str] = None,
        view_list: Optional[List[str]] = None,
    ):
        """
        :param image_height: resize 目标高（H）
        :param image_width: resize 目标宽（W）
        :param view: 旧版单视角参数；设置时优先于 view_list
        :param view_list: 按该顺序读取的相机字段名，默认 [cam_f0]
        """
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.view_list = normalize_image_view_list(view, view_list)
        # 保留旧属性，避免外部单视角调用读取 builder.view 时失效。
        self.view = self.view_list[0]

    def get_unique_name(self) -> str:
        # 单视角名称不能变化，确保已生成的 front_image_feature_v1_*.gz 可直接复用。
        if len(self.view_list) == 1:
            return (
                f"front_image_feature_v1_{self.view_list[0]}_"
                f"{self.image_height}x{self.image_width}"
            )
        views_tag = "-".join(self.view_list)
        return (
            f"front_image_feature_v2_v{len(self.view_list)}_{views_tag}_"
            f"{self.image_height}x{self.image_width}"
        )

    # ------------------------------------------------------------------ #
    # 离线缓存 / 在线 eval（有 scene）路径：复用 agent_input 取图逻辑
    # ------------------------------------------------------------------ #
    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        return self.compute_features(scene.get_agent_input())

    # ------------------------------------------------------------------ #
    # 从 agent_input 当前帧取一个或多个视角
    # ------------------------------------------------------------------ #
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        cameras = getattr(agent_input, "cameras", None)
        cam_bundle = cameras[-1] if cameras else None
        images = []
        valid = []
        for view in self.view_list:
            camera_obj = getattr(cam_bundle, view, None) if cam_bundle is not None else None
            image_field = getattr(camera_obj, "image", None) if camera_obj is not None else None
            arr = self._load_image(image_field)
            if arr is None:
                images.append(self._empty_image())
                valid.append(False)
            else:
                images.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
                valid.append(True)

        # V=1 时保留历史 (3,H,W) + scalar 契约，旧缓存和旧 collate 均无需迁移。
        if len(images) == 1:
            return {
                "front_image": images[0],
                "front_image_valid": torch.tensor(valid[0], dtype=torch.bool),
            }
        return {
            "front_image": torch.stack(images, dim=0),
            "front_image_valid": torch.tensor(valid, dtype=torch.bool),
        }

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _empty_feature_dict(self) -> Dict[str, torch.Tensor]:
        images = [self._empty_image() for _ in self.view_list]
        if len(images) == 1:
            return {
                "front_image": images[0],
                "front_image_valid": torch.tensor(False, dtype=torch.bool),
            }
        return {
            "front_image": torch.stack(images, dim=0),
            "front_image_valid": torch.zeros(len(images), dtype=torch.bool),
        }

    def _empty_image(self) -> torch.Tensor:
        return torch.zeros((3, self.image_height, self.image_width), dtype=torch.uint8)

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
            return np.array(img, dtype=np.uint8, copy=True)
        except Exception:
            return None


def collate_front_image_features(features: Dict[str, torch.Tensor], features_list) -> None:
    """把固定形状的图像张量 stack 进 batch。仅当缓存里带 front_image 时补。

    单视角得到 (B,3,H,W) / (B,)，多视角得到 (B,V,3,H,W) / (B,V)。与 hdmap
    变长特征不同，无需 pad_sequence。IL / RL / accelerate / ema 四处 collate 共用本函数。
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
