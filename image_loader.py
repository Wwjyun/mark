from pathlib import Path
from typing import Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QPixmap


def load_image(path: Path) -> Tuple[QPixmap, int, int]:
    """
    PIL → OpenCV (numpy) → QPixmap，全解析度載入。

    PIL 負責：格式相容性（TIFF、BMP、多種色彩空間）、EXIF 旋轉校正、16-bit 正規化。
    OpenCV 負責：色彩空間轉換（BGR→RGB），預留後續影像處理擴充點。
    QImage/QPixmap：最終顯示用。

    Returns:
        pixmap  – 供 QGraphicsScene 使用
        width   – 原始像素寬
        height  – 原始像素高
    """
    # --- 1. PIL 讀取 ---
    # 先用 pathlib 讀成 bytes，避免 Windows 中文路徑或打包後路徑編碼造成讀取失敗。
    try:
        with Image.open(BytesIO(path.read_bytes())) as img:
            pil = ImageOps.exif_transpose(img)   # 修正手機/相機 EXIF 旋轉
            pil.load()
    except Exception as exc:
        raise RuntimeError(f"無法讀取圖片：{path}\n{exc}") from exc

    # 統一轉成 uint8 RGB
    mode = pil.mode
    if mode in ("I", "I;16", "I;16B", "I;16L"):
        # 16-bit 灰階：線性壓縮至 8-bit（保留全部空間解析度）
        arr16 = np.array(pil, dtype=np.uint16)
        lo, hi = arr16.min(), arr16.max()
        if hi > lo:
            arr8 = ((arr16 - lo) * 255 / (hi - lo)).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr16, dtype=np.uint8)
        pil = Image.fromarray(arr8, mode="L").convert("RGB")
    elif mode == "RGBA":
        # 白底合成，去除透明通道
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[3])
        pil = bg
    elif mode != "RGB":
        pil = pil.convert("RGB")

    # --- 2. PIL → numpy (RGB) → OpenCV (BGR) ---
    arr_rgb: np.ndarray = np.array(pil, dtype=np.uint8)
    arr_bgr: np.ndarray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    # ── 預留 OpenCV 處理區 ──────────────────────────────────────────────
    # 例：arr_bgr = cv2.bilateralFilter(arr_bgr, 5, 75, 75)
    # ────────────────────────────────────────────────────────────────────

    # --- 3. OpenCV (BGR) → QImage (RGB) → QPixmap ---
    arr_out: np.ndarray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    arr_out = np.ascontiguousarray(arr_out)
    h, w = arr_out.shape[:2]

    # QImage 直接指向 numpy buffer；.copy() 確保 buffer 釋放後仍有效
    qimage = QImage(arr_out.data, w, h, w * 3, QImage.Format_RGB888).copy()

    return QPixmap.fromImage(qimage), w, h
