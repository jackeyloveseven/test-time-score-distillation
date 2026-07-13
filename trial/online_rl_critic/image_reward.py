"""Frozen CSD reward for the isolated same-state image-reward trial."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class OnlineCSDStyleReward:
    """Frozen CSD style similarity evaluated directly on decoded VAE tensors."""

    def __init__(
        self,
        *,
        style_path: str | Path,
        device: torch.device,
        csd_root: str = "/home/ubuntu/chengyanli/items/CSD_Score",
    ) -> None:
        self.device = torch.device(device)
        self.csd_root = Path(csd_root)
        if str(self.csd_root) not in sys.path:
            sys.path.insert(0, str(self.csd_root))
        from model import CSD_CLIP, convert_state_dict

        clip_weight = self.csd_root / "models" / "ViT-L-14.pt"
        checkpoint = self.csd_root / "models" / "checkpoint.pth"
        self.model = CSD_CLIP("vit_large", "default", model_path=str(clip_weight))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(convert_state_dict(state["model_state_dict"]), strict=False)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        self.style_embedding = self._embed_pil(Image.open(style_path).convert("RGB"))

    @torch.no_grad()
    def _embed_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        image_tensor = (image_tensor - self.mean) / self.std
        _, _, style = self.model(image_tensor)
        return F.normalize(style.float(), dim=-1)

    @torch.no_grad()
    def _embed_pil(self, image: Image.Image) -> torch.Tensor:
        tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        return self._embed_tensor(tensor)

    @torch.no_grad()
    def score_vae_output(self, vae_output: torch.Tensor) -> float:
        embedding = self._embed_tensor(vae_output)
        return float((embedding * self.style_embedding).sum(dim=-1).mean().item())

    def summary(self) -> dict[str, object]:
        return {
            "reward_model": "CSD_CLIP(vit_large, default)",
            "csd_root": str(self.csd_root),
            "style_embedding_shape": list(self.style_embedding.shape),
            "frozen": True,
        }
