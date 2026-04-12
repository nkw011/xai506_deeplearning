"""
Visualization utilities for DINOv2 semantic correspondence.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw

# Color palette for multiple points (BGR for cv2, then converted)
POINT_COLORS_RGB = [
    (255,  60,  60),  # red
    ( 60, 180, 255),  # blue
    ( 60, 220,  60),  # green
    (255, 200,  40),  # yellow
    (220,  60, 220),  # purple
    (255, 140,  40),  # orange
]


def apply_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Overlay a normalized [0,1] heatmap on a PIL image using jet colormap.

    Args:
        image: PIL Image (RGB).
        heatmap: float32 ndarray (H, W) with values in [0, 1].
        alpha: blend weight for the heatmap overlay.

    Returns:
        PIL Image with heatmap blended in.
    """
    img_np = np.array(image).astype(np.float32)

    # Jet colormap via cv2
    heat_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    blended = (1 - alpha) * img_np + alpha * heat_rgb
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def draw_correspondence_points(
    source_image: Image.Image,
    source_points: list[tuple[int, int]],
    target_image: Image.Image,
    best_matches: list[tuple[int, int]],
    radius: int = 10,
) -> tuple[Image.Image, Image.Image]:
    """Draw colored circles on source and target images for each point pair.

    Returns:
        (annotated_source, annotated_target)
    """
    src = source_image.copy()
    tgt = target_image.copy()
    src_draw = ImageDraw.Draw(src)
    tgt_draw = ImageDraw.Draw(tgt)

    for i, ((sx, sy), (tx, ty)) in enumerate(zip(source_points, best_matches)):
        color = POINT_COLORS_RGB[i % len(POINT_COLORS_RGB)]
        # Source point
        src_draw.ellipse(
            [sx - radius, sy - radius, sx + radius, sy + radius],
            fill=color, outline=(255, 255, 255), width=2,
        )
        src_draw.text((sx + radius + 3, sy - radius), str(i + 1),
                      fill=color)
        # Target best-match point
        tgt_draw.ellipse(
            [tx - radius, ty - radius, tx + radius, ty + radius],
            fill=color, outline=(255, 255, 255), width=2,
        )
        tgt_draw.text((tx + radius + 3, ty - radius), str(i + 1),
                      fill=color)

    return src, tgt


def make_heatmap_result(
    target_image: Image.Image,
    heatmap: np.ndarray,
    best_match: tuple[int, int],
    radius: int = 12,
) -> Image.Image:
    """Blend heatmap onto target image and mark the best-match point."""
    result = apply_heatmap(target_image, heatmap, alpha=0.55)
    draw = ImageDraw.Draw(result)
    tx, ty = best_match
    draw.ellipse(
        [tx - radius, ty - radius, tx + radius, ty + radius],
        fill=(255, 255, 0), outline=(0, 0, 0), width=2,
    )
    return result
