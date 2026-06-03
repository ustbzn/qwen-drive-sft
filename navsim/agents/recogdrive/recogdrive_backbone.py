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
    # questions：每个样本一条 prompt，和“一张相机图”一一对应；num_patches_list：每张图对应的 patch 数
    def forward_from_paths(self, image_paths: List[str], questions: List[str]):
        """Run the configured VLM from image paths and return hidden states."""
        if self.model_type == "qwen3":
            return self._forward_qwen3(image_paths=image_paths, questions=questions)

        pixel_values_list = [load_image(path) for path in image_paths]
        num_patches_list = [p.shape[0] for p in pixel_values_list]
        pixel_values_cat = torch.cat(pixel_values_list, dim=0).to(self.device)
        return self.forward(pixel_values_cat, questions, num_patches_list)

    def _forward_qwen3(self, image_paths: List[str], questions: List[str]):
        """Run Qwen3-VL using its processor/chat template and expose hidden states."""
        if self.processor is None:
            raise RuntimeError("Qwen3 processor is not initialized.")

        messages_batch = []
        for image_path, question in zip(image_paths, questions):
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
        model_device = next(self.model.parameters()).device
        inputs = inputs.to(model_device)
        return self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int]):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
        
        model_dtype = next(self.model.parameters()).dtype

        # 模型文本输入：queries
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
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

    
