"""Input and display helpers; the existing image algorithms remain in processor.py."""

import base64
import csv
import io
import warnings

from PIL import Image, UnidentifiedImageError

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 25_000_000
METHODS = {"orig": "Gốc", "p0": "P0 · HE", "p1": "P1 · CLAHE", "p2": "P2 · Pipeline"}
METRIC_COLUMNS = {
    "entropy": "Entropy (bit)",
    "std": "Độ tương phản (Std)",
    "sharpness": "Var Laplacian",
    "noise": "Nhiễu ước lượng",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "time_ms": "Thời gian (ms)",
}


def validate_image(data: bytes) -> tuple[int, int]:
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError("Vui lòng chọn ảnh không rỗng, tối đa 16 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "TIFF", "BMP", "WEBP"}:
                    raise ValueError("Chỉ hỗ trợ JPG, PNG, TIFF, BMP hoặc WebP; chưa hỗ trợ DICOM.")
                width, height = image.size
                if width * height > MAX_PIXELS:
                    raise ValueError("Ảnh quá lớn. Vui lòng dùng ảnh tối đa 25 triệu điểm ảnh.")
                image.verify()
                return width, height
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Không đọc được ảnh hoặc ảnh quá lớn. Hãy chọn một file ảnh hợp lệ.") from exc


def decode_image(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[1], validate=True)


def clamp_roi(x: int, y: int, size: int) -> tuple[int, int, int]:
    size = max(32, min(int(size), 512))
    return max(0, min(int(x), 512 - size)), max(0, min(int(y), 512 - size)), size


def roi_from_click(click: dict, size: int) -> tuple[int, int, int]:
    width = max(float(click.get("width", 512)), 1)
    height = max(float(click.get("height", 512)), 1)
    x = round(float(click["x"]) * 512 / width - size / 2)
    y = round(float(click["y"]) * 512 / height - size / 2)
    return clamp_roi(x, y, size)


def metric_rows(metrics: dict) -> list[dict]:
    return [
        {"Phương án": label, **{title: metrics[key][field] for field, title in METRIC_COLUMNS.items()}}
        for key, label in METHODS.items()
    ]


def metrics_csv(metrics: dict) -> bytes:
    rows = metric_rows(metrics)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")
