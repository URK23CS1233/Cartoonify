import cv2
import numpy as np


def cartoonify(image_bytes: bytes, style: str = "comic", num_colors: int = 8) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # Resize large images so NPR filters don't time out
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if style == "comic":
        result = _comic_style(img)
    elif style == "watercolor":
        result = _watercolor_style(img)
    elif style == "pencil":
        result = _pencil_style(img)
    else:
        result = _comic_style(img)

    _, buffer = cv2.imencode(".png", result)
    return buffer.tobytes()


def _comic_style(img: np.ndarray) -> np.ndarray:
    # Step 1: edge-preserving smooth — flattens colour regions cleanly
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)

    # Step 2: boost saturation so colours pop like a comic
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    smooth = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Step 3: strong black outlines via adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=7, C=3
    )
    # Thicken lines slightly
    edges = cv2.erode(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Step 4: burn edges onto smooth colour
    result = cv2.bitwise_and(smooth, edges_bgr)
    return result


def _watercolor_style(img: np.ndarray) -> np.ndarray:
    # cv2.stylization is OpenCV's built-in NPR watercolor-like filter
    # sigma_s controls spatial spread (0-200), sigma_r controls colour range (0-1)
    result = cv2.stylization(img, sigma_s=60, sigma_r=0.45)

    # Lighten slightly — watercolors are never fully saturated
    result = cv2.addWeighted(result, 0.85, np.full_like(result, 255), 0.15, 0)
    return result


def _pencil_style(img: np.ndarray) -> np.ndarray:
    # Divide trick: original ÷ blurred = fine pencil texture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    sketch = cv2.divide(gray, gray_blur, scale=256.0)

    # Optional: sharpen the sketch lines slightly
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sketch = cv2.filter2D(sketch, -1, kernel)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_bgr