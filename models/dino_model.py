"""
Grounding DINO wrapper for open-vocabulary object detection.
Model: IDEA-Research/grounding-dino-tiny
"""

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingDINODetector:
    DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.MODEL_ID = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load(self):
        print(f"[Grounding DINO] Loading {self.MODEL_ID} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        print("[Grounding DINO] Model loaded.")

    def predict(
        self,
        image: Image.Image,
        text_queries: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> tuple[list, list, list]:
        """Detect objects described by free-form text.

        Args:
            image: PIL Image (RGB).
            text_queries: Dot-separated category names, e.g. "a person. a laptop. a bag."
                          Each sentence is one category. Must end with a period.
            box_threshold: Minimum box confidence to keep a detection.
            text_threshold: Minimum text-label confidence.

        Returns:
            boxes:  List of [x1, y1, x2, y2] in pixel coords (xyxy).
            scores: List of float confidence scores.
            labels: List of string labels.
        """
        assert self.model is not None, "Call load() first."

        # Ensure the query ends with a period (required by Grounding DINO)
        query = text_queries.strip()
        if not query.endswith("."):
            query += "."

        inputs = self.processor(
            images=image,
            text=query,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results["labels"]

        return boxes, scores, labels
