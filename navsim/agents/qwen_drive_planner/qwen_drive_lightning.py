"""Qwen-Drive IL 的日志和 checkpoint 边界，冻结 VLM 从官方权重恢复。"""

from typing import Any, Dict

from navsim.planning.training.agent_lightning_module import AgentLightningModule


class QwenDriveLightningModule(AgentLightningModule):
    def _step(self, batch, logging_prefix):
        features, targets, tokens = batch
        prediction = self.agent(features, targets, tokens)
        for name in ("loss", "diffusion_loss", "temporal_loss"):
            self.log(
                f"{logging_prefix}/{name}",
                prediction[name],
                on_step=True,
                on_epoch=True,
                prog_bar=name == "loss",
                sync_dist=True,
                batch_size=targets["trajectory"].shape[0],
            )
        return prediction["loss"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items()
            if not key.startswith("agent._model.vlm.")
        }
        checkpoint["qwen_drive_vlm_path"] = self.agent.config.vlm_path

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        saved_vlm = checkpoint.get("qwen_drive_vlm_path")
        if saved_vlm is not None and saved_vlm != self.agent.config.vlm_path:
            raise ValueError("恢复训练必须使用 checkpoint 记录的同一 VLM 权重路径")
        checkpoint["state_dict"].update({
            key: value for key, value in self.state_dict().items()
            if key.startswith("agent._model.vlm.") and key not in checkpoint["state_dict"]
        })
