import base64
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from cartoonify import cartoonify
import os

app = FastAPI(title="Cartoonify API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/")
async def root():
    return FileResponse("index.html", media_type="text/html")


@app.post("/cartoonify")
async def cartoonify_image(
    file: UploadFile = File(...),
    style: str = Form(default="comic"),
    num_colors: int = Form(default=8),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP."
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    if style not in ("comic", "watercolor", "pencil"):
        style = "comic"

    num_colors = max(4, min(16, num_colors))

    try:
        result_bytes = cartoonify(image_bytes, style=style, num_colors=num_colors)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    original_b64 = base64.b64encode(image_bytes).decode("utf-8")
    result_b64 = base64.b64encode(result_bytes).decode("utf-8")

    original_mime = file.content_type
    return JSONResponse({
        "success": True,
        "style": style,
        "original": f"data:{original_mime};base64,{original_b64}",
        "cartoon": f"data:image/png;base64,{result_b64}",
        "original_size_kb": round(len(image_bytes) / 1024, 1),
        "cartoon_size_kb": round(len(result_bytes) / 1024, 1),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}