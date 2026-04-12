# XAI506 Deep Learning — Foundation Model Live Demo

Midterm assignment: live demo code using four Hugging Face foundation models.

---

## Models

| # | Task | Model | HuggingFace ID |
|---|------|-------|----------------|
| 1 | Point-based Segmentation | SAM | `facebook/sam-vit-base` |
| 2 | Open-Vocabulary Detection | Grounding DINO | `IDEA-Research/grounding-dino-base` |
| 3 | Visual Question Answering | Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` |

---

## Environment Setup

### 1. Install packages

```bash
conda create -n deeplearning python=3.11
pip install -r requirements.txt
```

> All other packages (`transformers`, `accelerate`, etc.) are listed in `requirements.txt`.

### 2. Pre-download model weights (required before presentation)

Download models in advance in case the venue has unreliable internet.

```bash
conda activate deeplearning

# SAM (~375 MB)
python -c "from transformers import SamModel, SamProcessor; SamModel.from_pretrained('facebook/sam-vit-base')"

# Grounding DINO (~340 MB)
python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')"

# Qwen2-VL (~15 GB, takes time)
python -c "from transformers import Qwen2VLForConditionalGeneration; Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', torch_dtype='auto')"

# DINOv2 (~330 MB)
python -c "from transformers import AutoModel, AutoImageProcessor; AutoModel.from_pretrained('facebook/dinov2-base')"
```
---

## How to Use Each Jupyter Notebook

Each demo has a standalone notebook for step-by-step exploration.

| Notebook | Demo | Description |
|----------|------|-------------|
| [notebook_sam.ipynb](notebook_sam.ipynb) | SAM | Load image → click points interactively → run segmentation |
| [notebook_grounding_dino.ipynb](notebook_grounding_dino.ipynb) | Grounding DINO | Load image → enter text query → detect and visualize objects |
| [notebook_vlm.ipynb](notebook_vlm.ipynb) | Qwen2-VL | Load image → auto caption → single-turn Q&A → multi-turn conversation |

---

## Project Structure

```
xai506_deeplearning/
├── models/
│   ├── sam_model.py              # SAMSegmentor class
│   ├── dino_model.py             # GroundingDINODetector class
│   ├── vlm_model.py              # Qwen2VLQA class
│   └── dino_correspondence.py    # DINOCorrespondence class
├── utils/
│   ├── visualization.py          # Mask / point / box visualization utilities
│   └── correspondence_vis.py     # Heatmap and correspondence visualization utilities
└── requirements.txt
```

---
