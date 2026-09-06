from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ClipVisualTextEncoder:
    """Optional OpenCLIP adapter. Importing the app does not download model weights."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Install optional retrieval dependencies with "
                "`python -m pip install -r requirements_retrieval.txt`."
            ) from exc
        self.torch = torch
        self.open_clip = open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def score(self, text: str, image_path: Path) -> float:
        if not image_path.exists():
            raise FileNotFoundError(f"Image is missing: {image_path}")
        with Image.open(image_path) as image, self.torch.no_grad():
            image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            text_tensor = self.tokenizer([text]).to(self.device)
            image_feature = self.model.encode_image(image_tensor)
            text_feature = self.model.encode_text(text_tensor)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            cosine = float((image_feature @ text_feature.T).item())
        # The UI combines scores on [0, 1]. This is a display normalisation, not probability calibration.
        return (cosine + 1.0) / 2.0
