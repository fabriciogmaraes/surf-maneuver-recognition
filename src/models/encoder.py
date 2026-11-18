"""Frozen V-JEPA 2 video encoder: frames in, one feature vector per time slice out."""
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2Encoder:
    """Load the encoder once, then call it on clips of 16 RGB frames."""

    def __init__(self, model_id=MODEL_ID, device="cuda"):
        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        # float16 halves the memory use; the RTX 3050 has 6 GB
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """(16, H, W, 3) uint8 RGB frames -> (8, 1024): tokens averaged over space for each time slice."""
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)   # (T, C, H, W)
        inputs = self.processor(video, return_tensors="pt").to(device=self.device, dtype=torch.float16)
        tokens = self.model.get_vision_features(**inputs)                            # (1, 2048, 1024)
        # Tokens are ordered by time, then space: 8 time slices x 256 patches
        return tokens.reshape(8, 256, -1).mean(dim=1).float().cpu().numpy()