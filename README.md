# XAI506 Deep Learning — Foundation Model Live Demo

Midterm assignment: live demo code using four Hugging Face foundation models.

---

## Models

| # | Task | Model | HuggingFace ID |
|---|------|-------|----------------|
| 1 | Point-based Segmentation | SAM | `facebook/sam-vit-base` |
| 2 | Open-Vocabulary Detection | Grounding DINO | `IDEA-Research/grounding-dino-base` |
| 3 | Visual Question Answering | Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` |
| 4 | Semantic Correspondence | DINOv2 | `facebook/dinov2-base` |

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
| [notebook_correspondence.ipynb](notebook_correspondence.ipynb) | DINOv2 | Load source/target images → click source → visualize similarity heatmap |


### Tab 1 — Point-based Segmentation (SAM)

1. Upload a photo.
2. Select `Foreground (include)` or `Background (exclude)` under **Point Label**.
3. Click on the image to add points.
   - Red dot = Foreground (region to include)
   - Blue dot = Background (region to exclude)
4. Use **Undo** / **Clear** to modify points.
5. Press **Segment!** to see the result.

### Tab 2 — Open-Vocabulary Detection (Grounding DINO)

1. Upload a photo.
2. Enter categories to detect in **Text Query**, separated by periods.  
   Example: `a person. a face. a laptop.`
3. Adjust **Box Threshold** / **Text Threshold** sliders as needed.
4. Press **Detect!** to see the result.

> **Tip:** Try changing only the query on the same image for an effective demo.  
> e.g., `a person.` → `a glasses.` → `a shirt.`

### Tab 3 — Visual Question Answering (Qwen2-VL)

1. Upload a photo.
2. Click **Auto Caption** to generate an automatic image description.
3. Or type a question in the input box and press **Ask**.
4. You can ask multiple questions about the same image (multi-turn conversation supported).
5. Use **Clear Chat** to reset the conversation.

**Example questions:**
- `What is this person wearing?`
- `How many people are in this image?`
- `Is the person smiling?`
- `What are the students doing in this classroom?`

### Tab 4 — Semantic Correspondence (DINOv2)

1. Upload a **Source** image and a **Target** image.
2. Click on the Source image to add query points.
3. Press **Find Correspondence!** to see the similarity heatmap on the Target image.
4. Use **Undo** / **Clear Points** to modify selected points.

---

```bash
conda activate deeplearning
jupyter lab
```

> The notebooks use `%matplotlib widget` for interactive clicking — make sure `ipympl` is installed (`pip install ipympl`).

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
