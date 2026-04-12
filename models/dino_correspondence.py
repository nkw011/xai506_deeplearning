"""
DINOv2 semantic correspondence wrapper.
Model: facebook/dinov2-base

Given a click point on a source image, computes cosine similarity between
that patch feature and every patch in the target image, returning a heatmap.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOCorrespondence:
    MODEL_ID = "facebook/dinov2-base"
    PATCH_SIZE = 14   # DINOv2 uses 14×14 patches

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load(self):
        print(f"[DINOv2] Loading {self.MODEL_ID} on {self.device} ...")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        print("[DINOv2] Model loaded.")

    def _extract_features(self, image: Image.Image) -> tuple[torch.Tensor, int, int]:
        """Extract patch feature grid from a PIL image.

        Returns:
            features: float32 tensor  (h_patches, w_patches, dim)
            h_patches, w_patches: grid dimensions
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        ph = inputs["pixel_values"].shape[2]
        pw = inputs["pixel_values"].shape[3]
        h_patches = ph // self.PATCH_SIZE
        w_patches = pw // self.PATCH_SIZE

        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(**inputs)

        # last_hidden_state: (1, 1 + h*w, dim)  — first token is [CLS]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (1, h*w, dim)
        features = patch_tokens.reshape(h_patches, w_patches, -1).float()  # (h, w, dim)
        return features, h_patches, w_patches

    def find_correspondence(
        self,
        source_image: Image.Image,
        source_point: tuple[int, int],
        target_image: Image.Image,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Compute similarity heatmap between one source point and all target patches.

        Args:
            source_image: PIL Image (RGB).
            source_point: (x, y) pixel coordinate clicked on source_image.
            target_image: PIL Image (RGB).

        Returns:
            heatmap: float32 ndarray (H_target, W_target) with values in [0, 1].
            best_match: (x, y) pixel coordinate of best match in target_image.
        """
        assert self.model is not None, "Call load() first."

        src_feat, src_h, src_w = self._extract_features(source_image)
        tgt_feat, tgt_h, tgt_w = self._extract_features(target_image)

        # Map source click pixel → patch grid index
        sx, sy = source_point
        px = min(int(sx / source_image.width  * src_w), src_w - 1)
        py = min(int(sy / source_image.height * src_h), src_h - 1)

        query = src_feat[py, px]                          # (dim,)
        query = F.normalize(query.unsqueeze(0), dim=-1)   # (1, dim)

        # Cosine similarity with all target patches
        tgt_flat = tgt_feat.reshape(-1, tgt_feat.shape[-1])          # (h*w, dim)
        tgt_flat = F.normalize(tgt_flat, dim=-1)
        sim = (query @ tgt_flat.T).squeeze(0).reshape(tgt_h, tgt_w)  # (h, w)
        sim_np = sim.cpu().numpy()

        # Upsample to target image resolution
        heatmap = cv2.resize(sim_np, (target_image.width, target_image.height),
                             interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Best match pixel
        flat_idx = sim_np.argmax()
        best_py, best_px = divmod(int(flat_idx), tgt_w)
        best_x = int((best_px + 0.5) / tgt_w  * target_image.width)
        best_y = int((best_py + 0.5) / tgt_h * target_image.height)

        return heatmap, (best_x, best_y)

    def find_multi_correspondence(
        self,
        source_image: Image.Image,
        source_points: list[tuple[int, int]],
        target_image: Image.Image,
    ) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        """Run find_correspondence for multiple source points efficiently.

        Extracts features once per image, then queries each point.
        """
        assert self.model is not None, "Call load() first."

        src_feat, src_h, src_w = self._extract_features(source_image)
        tgt_feat, tgt_h, tgt_w = self._extract_features(target_image)

        tgt_flat = F.normalize(tgt_feat.reshape(-1, tgt_feat.shape[-1]), dim=-1)

        heatmaps, best_matches = [], []
        for sx, sy in source_points:
            px = min(int(sx / source_image.width  * src_w), src_w - 1)
            py = min(int(sy / source_image.height * src_h), src_h - 1)

            query = F.normalize(src_feat[py, px].unsqueeze(0), dim=-1)
            sim = (query @ tgt_flat.T).squeeze(0).reshape(tgt_h, tgt_w).cpu().numpy()

            heatmap = cv2.resize(sim, (target_image.width, target_image.height),
                                 interpolation=cv2.INTER_LINEAR)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            flat_idx = sim.argmax()
            bpy, bpx = divmod(int(flat_idx), tgt_w)
            best_x = int((bpx + 0.5) / tgt_w  * target_image.width)
            best_y = int((bpy + 0.5) / tgt_h * target_image.height)

            heatmaps.append(heatmap)
            best_matches.append((best_x, best_y))

        return heatmaps, best_matches
