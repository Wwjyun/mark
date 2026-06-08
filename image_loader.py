from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QPixmap


# Pillow defaults to rejecting very large images as a decompression-bomb guard.
# This tool is local annotation software, so large industrial/camera images are expected.
Image.MAX_IMAGE_PIXELS = None


def load_image(path: Path) -> Tuple[QPixmap, int, int]:
    """
    Load an image into a QPixmap at full resolution.

    The file is read through pathlib first to avoid Windows unicode path issues.
    The conversion path intentionally avoids OpenCV RGB/BGR round trips because
    those create extra full-size image copies and can fail on large images.
    """
    try:
        with Image.open(BytesIO(path.read_bytes())) as img:
            pil = ImageOps.exif_transpose(img)
            pil.load()
    except Exception as exc:
        raise RuntimeError(f"無法讀取圖片：{path}\n{exc}") from exc

    mode = pil.mode
    if mode in ("I", "I;16", "I;16B", "I;16L"):
        arr16 = np.asarray(pil, dtype=np.uint16)
        lo, hi = arr16.min(), arr16.max()
        if hi > lo:
            arr8 = ((arr16 - lo) * 255 / (hi - lo)).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr16, dtype=np.uint8)
        pil = Image.fromarray(arr8, mode="L").convert("RGB")
    elif mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[3])
        pil = bg
    elif mode != "RGB":
        pil = pil.convert("RGB")

    arr_rgb = np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))
    h, w = arr_rgb.shape[:2]

    qimage = QImage(arr_rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage), w, h
