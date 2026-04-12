"""
SAM (Segment Anything Model) wrapper for point-based segmentation.
Model: facebook/sam-vit-base
"""

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


class SAMSegmentor:
    MODEL_ID = "facebook/sam-vit-base"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: SamModel | None = None
        self.processor: SamProcessor | None = None

    def load(self):
        print(f"[SAM] Loading {self.MODEL_ID} on {self.device} ...")
        self.processor = SamProcessor.from_pretrained(self.MODEL_ID)
        self.model = SamModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        print("[SAM] Model loaded.")

    def predict(
        self,
        image: Image.Image,
        points: list[list[int]],
        labels: list[int],
    ) -> tuple[np.ndarray, float]:
        """Run SAM with click points.

        Args:
            image: PIL Image (RGB).
            points: List of [x, y] pixel coordinates.
            labels: List of 1 (foreground) or 0 (background) per point.

        Returns:
            best_mask: Boolean ndarray of shape (H, W).
            score: IoU score of the selected mask.
        """
        assert self.model is not None, "Call load() first."

        inputs = self.processor(
            images=image,
            input_points=[points],   # shape: (1, N, 2)
            input_labels=[labels],   # shape: (1, N)
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(**inputs)

        # post_process_masks returns list[list[Tensor]]  shape: (1, 1, 3, H, W)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0][0] shape: (3, H, W) – 3 candidate masks per point set
        iou_scores = outputs.iou_scores[0, 0]  # shape: (3,)
        best_idx = int(iou_scores.argmax())
        best_mask = masks[0][0][best_idx].numpy().astype(bool)  # (H, W)
        return best_mask, float(iou_scores[best_idx])
