"""用一个真实 Mogo 样本验证在线 VLM、官方规划头和一次 IL 参数更新。"""

import argparse
import json
import pickle
import time
from pathlib import Path

import torch

from navsim.agents.qwen_drive_planner.qwen_drive_agent import QwenDriveAgent
from navsim.common.dataclasses import Scene
from navsim.planning.script.run_training_qwen_drive_il import custom_collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offload-vlm", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    device = torch.device(args.device)
    started = time.monotonic()

    log_path = args.data_root / "navsim_logs" / "train" / (args.log.removesuffix(".pkl") + ".pkl")
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)[args.start_frame:args.start_frame + 66]
    if len(frames) != 66:
        raise ValueError("该起点不能提供 16 个历史帧和 50 个未来帧")
    agent = QwenDriveAgent()
    scene = Scene.from_scene_dict_list(
        frames, args.data_root / "sensor_blobs" / "train",
        num_history_frames=16, num_future_frames=50,
        sensor_config=agent.get_sensor_config(), load_image_path=True,
    )
    feature = agent.get_feature_builders()[0].compute_features(scene.get_agent_input())
    target = agent.get_target_builders()[0].compute_targets(scene)
    features, targets, tokens = custom_collate_fn([(feature, target)])
    print(json.dumps({"phase": "data", "target_shape": list(targets["trajectory"].shape)}), flush=True)

    agent.initialize()
    agent.to(device).train()
    model = agent._model
    cache_shapes = []
    encode_scenes = agent._encode_scenes

    def measured_encode(batch):
        if args.offload_vlm:
            model.planning_expert.cpu()
            torch.cuda.empty_cache()
        caches, anchor, valid = encode_scenes(batch)
        cache_shapes.extend([list(key.shape) for key, _ in caches])
        if args.offload_vlm:
            model.vlm.cpu()
            torch.cuda.empty_cache()
            model.planning_expert.to(device)
        print(json.dumps({
            "phase": "vlm", "cache_shapes": cache_shapes,
            "anchor": anchor.tolist(), "valid_tokens": valid.sum().item(),
        }), flush=True)
        return caches, anchor, valid

    agent._encode_scenes = measured_encode
    optimizer = agent.get_optimizers()
    before = model.planning_expert.out_proj.weight.detach().clone()
    prediction = agent(features, targets, tokens)
    assert torch.isfinite(prediction["loss"])
    prediction["loss"].backward()
    parameters = list(model.planning_expert.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    assert all(parameter.grad is None for parameter in model.vlm.parameters())
    assert model.planning_expert.layers[0].qkv_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    delta = (model.planning_expert.out_proj.weight.detach() - before).float().abs().max()
    assert delta > 0
    print(json.dumps({
        "phase": "complete", "loss": prediction["loss"].item(),
        "diffusion_loss": prediction["diffusion_loss"].item(),
        "temporal_loss": prediction["temporal_loss"].item(),
        "gradient_norm": grad_norm.item(), "head_weight_max_delta": delta.item(),
        "vlm_has_grad": False, "prediction_shape": list(prediction["trajectory"].shape),
        "offload_vlm": args.offload_vlm, "elapsed_seconds": time.monotonic() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
