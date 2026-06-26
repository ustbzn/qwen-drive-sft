from pathlib import Path
from typing import List, Optional, Tuple, Union
import json
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils.conversation import get_conv_template
from .utils.internvl_preprocess import load_image

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


def load_qwen3_processor(checkpoint_path: str):
    """Load Qwen3 processor, falling back around local fast-tokenizer config issues."""
    try:
        return AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
    except AttributeError as exc:
        if "model_type" not in str(exc):
            raise

    model_path = Path(checkpoint_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint_path)
    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    video_processor = Qwen3VLVideoProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )

    chat_template = None
    chat_template_path = model_path / "chat_template.json"
    if chat_template_path.is_file():
        with chat_template_path.open("r", encoding="utf-8") as f:
            chat_template = json.load(f).get("chat_template")

    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )


class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_type = model_type.lower()
        self.device = device
        self.vl_hidden_dim = None

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256
            llm_config = getattr(self.model.config, "llm_config", None)
            self.vl_hidden_dim = int(getattr(llm_config, "hidden_size", 1536))

        elif self.model_type == 'qwen':
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path,
                trust_remote_code=True
            )
            self.vl_hidden_dim = int(self.model.config.text_config.hidden_size)

        elif self.model_type == 'qwen3':
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            ).eval()
            self.processor = load_qwen3_processor(checkpoint_path)
            self.vl_hidden_dim = int(self.model.config.text_config.hidden_size)
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl', 'qwen', or 'qwen3'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    # 把图像 + 文本 prompt喂给 VLM（InternVL/Qwen），输出 transformer 的 hidden states
    # 输入是 [batch][view][frame] 三层路径矩阵（每样本一段多视角/多帧 clip）。
    # 单视角单帧时 view_frame_paths = [[[front_path]]]，与改造前行为等价。
    @staticmethod
    def _normalize_view_frame_paths(view_frame_paths):
        """把可能的旧式扁平输入规整为 [batch][view][frame] 三层结构。

        兼容历史调用：传 List[str]（一 batch 一图）时包装成
        [[[p]] for p in paths]；传 [batch][view][frame] 时原样返回。
        """
        if not view_frame_paths:
            return view_frame_paths
        sample0 = view_frame_paths[0]
        if isinstance(sample0, str):
            # List[str]：每个元素是一个样本的单视角单帧路径
            return [[[p]] for p in view_frame_paths]
        return view_frame_paths

    def forward_from_paths(
        self,
        view_frame_paths,
        questions: List[str],
        *,
        view_labels: Optional[List[str]] = None,
        num_frames: int = 1,
        use_video_input: bool = False,
        video_fps: float = 2.0,
    ):
        """Run the configured VLM from image paths and return hidden states.

        :param view_frame_paths: [batch][view][frame] 图像路径矩阵。
        :param questions: 每样本一条 prompt。
        :param view_labels: 长度 = n_views 的视角标签（如 ['FRONT VIEW', ...]）。
        :param num_frames: 每视角帧数；>1 且 use_video_input 时 Qwen3 走视频路径。
        :param use_video_input: 仅 Qwen3 生效；True 且 num_frames>1 时用 videos= 路径。
        :param video_fps: VideoMetadata 用。
        """
        view_frame_paths = self._normalize_view_frame_paths(view_frame_paths)

        if self.model_type == "qwen3":
            if use_video_input and int(num_frames) > 1:
                return self._forward_qwen3_video(
                    view_frame_paths, questions,
                    view_labels=view_labels, video_fps=video_fps,
                )
            return self._forward_qwen3(
                view_frame_paths, questions,
                view_labels=view_labels, num_frames=num_frames,
            )

        # InternVL：把 [batch][view][frame] 扁平成图像列表，逐图计算 num_patches。
        if self.model_type == "internvl" and use_video_input:
            print("[RecogDriveBackbone] InternVL does not support video input; "
                  "falling back to multi-image frames.")
        flat_paths, per_sample_counts = [], []
        for sample_views in view_frame_paths:
            cnt = 0
            for frames in sample_views:
                for p in frames:
                    flat_paths.append(p)
                    cnt += 1
            per_sample_counts.append(cnt)
        pixel_values_list = [load_image(path) for path in flat_paths]
        num_patches_list = [p.shape[0] for p in pixel_values_list]
        pixel_values_cat = torch.cat(pixel_values_list, dim=0).to(self.device)
        return self.forward(
            pixel_values_cat, questions, num_patches_list,
            per_sample_image_counts=per_sample_counts,
        )

    def _forward_qwen3(self, view_frame_paths, questions: List[str], *,
                       view_labels: Optional[List[str]] = None, num_frames: int = 1):
        """Run Qwen3-VL via processor and expose hidden states.

        Covers single-view single-frame (现状，零影响) and multi-view single-frame；
        多帧但未开 video 时把每帧当独立 image（带 Frame-i 标签）。
        """
        from PIL import Image

        if self.processor is None:
            raise RuntimeError("Qwen3 processor is not initialized.")

        n_views = len(view_frame_paths[0]) if view_frame_paths else 1
        multi = (n_views > 1) or (int(num_frames) > 1)

        # 单视角单帧：逐字复刻改造前的调用（message 直接带 image 路径 +
        # apply_chat_template(tokenize=True)），保证 hidden_state 数值零影响。
        if not multi:
            messages_batch = []
            for sample_views, question in zip(view_frame_paths, questions):
                image_path = sample_views[0][0]
                messages_batch.append([
                    {"role": "system", "content": [{"type": "text", "text": system_message}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question},
                        ],
                    },
                ])
            inputs = self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(next(self.model.parameters()).device)
            return self.model(**inputs, output_hidden_states=True, return_dict=True)

        # 多视角 / 多帧：tokenize=False 出文本 + processor(images=PIL列表) 统一上图。
        messages_batch, pil_images = [], []
        for sample_views, question in zip(view_frame_paths, questions):
            user_content = []
            for v_idx, frames in enumerate(sample_views):
                label = view_labels[v_idx] if view_labels else ""
                for f_idx, p in enumerate(frames):
                    suffix = f" Frame-{f_idx + 1}" if int(num_frames) > 1 else ""
                    user_content.append({"type": "text", "text": f"{label}{suffix}:"})
                    user_content.append({"type": "image"})
                    pil_images.append(Image.open(p).convert("RGB"))
            user_content.append({"type": "text", "text": question})
            messages_batch.append([
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": user_content},
            ])

        chat_texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = self.processor(
            text=chat_texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(next(self.model.parameters()).device)
        return self.model(**inputs, output_hidden_states=True, return_dict=True)

    def _forward_qwen3_video(self, view_frame_paths, questions: List[str], *,
                             view_labels: Optional[List[str]] = None, video_fps: float = 2.0):
        """Run Qwen3-VL with each camera as one video clip (multi-frame temporal).

        移植 DrivePi_8GPU 的 video 组装：每视角一段 [T,H,W,C] uint8 clip，按
        (sample, view) flatten 成 videos 列表 + 一一对应的 VideoMetadata，并显式传
        do_sample_frames=False 禁止 processor 重采样。
        """
        import numpy as np
        from PIL import Image
        from transformers.video_utils import VideoMetadata

        if self.processor is None:
            raise RuntimeError("Qwen3 processor is not initialized.")

        videos, video_metadata = [], []
        for sample_views in view_frame_paths:            # batch
            for frames in sample_views:                  # 每视角一段 clip（旧→新）
                arr = np.stack(
                    [np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in frames],
                    axis=0,
                )  # [T, H, W, C]
                t_, h_, w_ = arr.shape[0], arr.shape[1], arr.shape[2]
                videos.append(arr)
                video_metadata.append(VideoMetadata(
                    total_num_frames=t_,
                    fps=video_fps,
                    width=w_,
                    height=h_,
                    duration=t_ / float(video_fps),
                    video_backend="decord",
                    frames_indices=list(range(t_)),
                ))

        n_views = len(view_frame_paths[0]) if view_frame_paths else 1
        chat_texts = []
        for question in questions:
            content = []
            for v_idx in range(n_views):
                label = view_labels[v_idx] if view_labels else ""
                content.append({"type": "text", "text": f"{label}:"})
                content.append({"type": "video"})
            content.append({"type": "text", "text": question})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": content},
            ]
            chat_texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        inputs = self.processor(
            text=chat_texts,
            videos=videos,
            video_metadata=video_metadata,
            do_sample_frames=False,  # 已是 pre-sampled 帧，禁止 processor 再按 fps 重采样
            return_tensors="pt",
            padding=True,
        ).to(next(self.model.parameters()).device)
        return self.model(**inputs, output_hidden_states=True, return_dict=True)

    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int],
                per_sample_image_counts: Optional[List[int]] = None):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")

        model_dtype = next(self.model.parameters()).dtype

        # per_sample_image_counts：每个样本（question）含几张图。缺省视作一图一问
        # （单视角单帧的历史行为）。多视角时一个 question 对应多张图，需按此切分
        # num_patches_list，并把该样本 prompt 里的每个 <image> 占位逐个替换为对应图的
        # IMG token（修复原先 replace(..., 1) 只替换第一个占位的 bug）。
        if per_sample_image_counts is None:
            per_sample_image_counts = [1] * len(questions)

        # 模型文本输入：queries
        queries = []
        img_cursor = 0
        for s_idx, question in enumerate(questions):
            n_img = per_sample_image_counts[s_idx]
            sample_patches = num_patches_list[img_cursor:img_cursor + n_img]
            img_cursor += n_img

            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question

            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            # 按出现顺序逐个替换 <image>，每个占位用各自图的 num_patches。
            parts = query.split('<image>')
            n_placeholder = len(parts) - 1
            if n_placeholder != len(sample_patches):
                raise ValueError(
                    f"InternVL prompt has {n_placeholder} <image> placeholders but "
                    f"{len(sample_patches)} images for sample {s_idx}."
                )
            rebuilt = parts[0]
            for k, np_k in enumerate(sample_patches):
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * np_k + IMG_END_TOKEN
                rebuilt += image_tokens + parts[k + 1]
            queries.append(rebuilt)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(
            queries,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=2800,
        )  # tokenizer 把 queries 转成模型要的 input_ids 和 attention_mask，放到 GPU 上
        device = torch.device('cuda')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        num_patches = pixel_values.size(0)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)


        return self.model(
                pixel_values=pixel_values.to(model_dtype),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
        )

    
