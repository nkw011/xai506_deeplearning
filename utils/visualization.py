"""
Shared visualization utilities using PIL.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Color palette for multiple detections
COLORS = [
    (255, 0, 0),    # red
    (0, 200, 0),    # green
    (0, 100, 255),  # blue
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (0, 200, 200),  # cyan
    (255, 20, 147), # deep pink
    (255, 215, 0),  # gold
]


def draw_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple = (0, 200, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay a boolean mask on a PIL image with transparency.

    Args:
        image: Input PIL image (RGB).
        mask: Boolean numpy array of shape (H, W).
        color: RGB tuple for mask color.
        alpha: Opacity of the mask overlay (0=transparent, 1=opaque).

    Returns:
        PIL image with mask overlaid.
    """
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_rgba[mask] = (*color, int(255 * alpha))

    overlay_from_array = Image.fromarray(mask_rgba, mode="RGBA")
    composite = Image.alpha_composite(img, overlay_from_array)
    return composite.convert("RGB")


def draw_points_on_image(
    image: Image.Image,
    points: list,
    labels: list,
    radius: int = 8,
) -> Image.Image:
    """Draw click points on image. Foreground=red, Background=blue.

    Args:
        image: Input PIL image.
        points: List of [x, y] coordinates.
        labels: List of int (1=foreground, 0=background).
        radius: Circle radius in pixels.

    Returns:
        PIL image with points drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for (x, y), label in zip(points, labels):
        fill = (255, 50, 50) if label == 1 else (50, 100, 255)
        outline = (200, 0, 0) if label == 1 else (0, 0, 200)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=fill,
            outline=outline,
            width=2,
        )
    return img


def draw_boxes_on_image(
    image: Image.Image,
    boxes: list,
    scores: list,
    labels: list,
    line_width: int = 3,
) -> Image.Image:
    """Draw bounding boxes with labels on a PIL image.

    Args:
        image: Input PIL image.
        boxes: List/tensor of [x1, y1, x2, y2] in pixel coords.
        scores: List/tensor of confidence scores.
        labels: List of string labels.
        line_width: Box border width.

    Returns:
        PIL image with boxes drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in box]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background + text
        text = f"{label} {float(score):.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_h = bbox[3] - bbox[1] + 4
        text_w = bbox[2] - bbox[0] + 4
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h + 2), text, fill=(255, 255, 255), font=font)

    return img
