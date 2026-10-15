"""
processor.py - Module xử lý ảnh y khoa X-quang đa vùng xương
Đề tài: Tăng cường chất lượng hiển thị ảnh X-quang đa vùng xương (Fracture Multi-Region X-ray)

Bao gồm:
- Chuẩn hóa dải động Min-Max và resize 512x512
- P0: Global Histogram Equalization
- P1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- P2: Proposed Pipeline (Lọc nhiễu -> CLAHE -> Unsharp Masking)
- Phát hiện biên Sobel & Canny, Overlay biên màu đỏ (#FF3B30)
- Trích xuất Zoom ROI & vẽ bounding box
- Tính toán 6 chỉ số định lượng (Entropy, Std, Var Laplacian, Noise, PSNR, SSIM) & thời gian chạy ms
- Tính histogram 256 mức xám
"""

import time
import base64
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import shannon_entropy


def encode_image_to_base64(img_array: np.ndarray, is_rgb: bool = False, format_ext: str = '.png') -> str:
    """
    Chuyển đổi numpy array ảnh (Grayscale hoặc RGB) thành chuỗi Data URL Base64.
    """
    if img_array is None or img_array.size == 0:
        return ""
    
    if is_rgb:
        # OpenCV imencode cần định dạng BGR
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        bgr_img = img_array

    encode_params = []
    if format_ext == '.jpg' or format_ext == '.jpeg':
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"

    success, buffer = cv2.imencode(format_ext, bgr_img, encode_params)
    if not success:
        raise ValueError("Lỗi mã hóa ảnh sang Base64.")
    
    encoded_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_str}"


def read_and_normalize_image(image_bytes: bytes, target_size: int = 512) -> np.ndarray:
    """
    Đọc dữ liệu byte ảnh, chuyển sang Grayscale, chuẩn hóa Min-Max về uint8 [0, 255]
    và resize về target_size x target_size.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Đọc ảnh giữ nguyên độ sâu nếu có (hoặc grayscale)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img_raw is None:
        raise ValueError("Không thể giải mã dữ liệu ảnh. Vui lòng kiểm tra định dạng file.")

    # Xử lý kênh màu nếu ảnh có nhiều kênh
    if len(img_raw.shape) == 3:
        if img_raw.shape[2] == 4:
            # RGBA
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2GRAY)
        else:
            # BGR
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_raw

    # Chuẩn hóa Min-Max dải động về uint8 [0, 255]
    gray_float = gray.astype(np.float64)
    min_val = np.min(gray_float)
    max_val = np.max(gray_float)

    if max_val > min_val:
        normalized = ((gray_float - min_val) / (max_val - min_val) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(gray, dtype=np.uint8)

    # Resize về target_size x target_size (512x512)
    h, w = normalized.shape[:2]
    if (w, h) != (target_size, target_size):
        interp = cv2.INTER_AREA if (w > target_size or h > target_size) else cv2.INTER_CUBIC
        normalized = cv2.resize(normalized, (target_size, target_size), interpolation=interp)

    return normalized


def compute_metrics(original: np.ndarray, processed: np.ndarray, exec_time_ms: float, is_reference: bool = False) -> dict:
    """
    Tính 6 chỉ số định lượng:
    1. Entropy (Shannon Entropy - bits)
    2. Std (Độ tương phản RMS)
    3. Variance of Laplacian (Độ sắc nét biên)
    4. Noise (Độ nhiễu tần số cao ước lượng = std(img - Gaussian3x3))
    5. PSNR (dB) so với ảnh gốc
    6. SSIM so với ảnh gốc
    Kèm thời gian thực thi (ms) và histogram 256 mức.
    """
    # 1. Entropy
    ent = float(shannon_entropy(processed))

    # 2. RMS Contrast (Std)
    rms_contrast = float(np.std(processed))

    # 3. Variance of Laplacian (Độ sắc nét)
    lap = cv2.Laplacian(processed, cv2.CV_64F)
    sharpness = float(lap.var())

    # 4. High-frequency Noise estimation
    smoothed = cv2.GaussianBlur(processed, (3, 3), 0)
    noise_residual = processed.astype(np.float64) - smoothed.astype(np.float64)
    noise_est = float(np.std(noise_residual))

    # 5. PSNR & 6. SSIM
    if is_reference:
        psnr_val = None
        ssim_val = 1.0000
    else:
        try:
            psnr_val = float(peak_signal_noise_ratio(original, processed, data_range=255))
        except Exception:
            psnr_val = 0.0

        try:
            ssim_val = float(structural_similarity(original, processed, data_range=255))
        except Exception:
            ssim_val = 0.0

    # 7. Histogram 256 bins
    hist = cv2.calcHist([processed], [0], None, [256], [0, 256]).flatten()
    # Normalize hist for easier graph reading if desired, or keep raw counts
    hist_list = [int(v) for v in hist]

    return {
        "entropy": round(ent, 4),
        "std": round(rms_contrast, 2),
        "sharpness": round(sharpness, 2),
        "noise": round(noise_est, 2),
        "psnr": round(psnr_val, 2) if psnr_val is not None else "Ref (Gốc)",
        "ssim": round(ssim_val, 4),
        "time_ms": round(exec_time_ms, 2),
        "hist": hist_list
    }


def compute_sobel_and_canny(img: np.ndarray, canny_low: int = 50, canny_high: int = 150):
    """
    Tính ảnh biên Sobel Magnitude và biên Canny.
    """
    # Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    sobel_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Canny
    canny = cv2.Canny(img, canny_low, canny_high)

    return sobel_norm, canny


def create_red_overlay(base_img: np.ndarray, canny_mask: np.ndarray, color_rgb=(255, 59, 48)) -> np.ndarray:
    """
    Tạo ảnh RGB với biên Canny màu đỏ (#FF3B30) phủ lên ảnh xám base_img.
    """
    rgb_base = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    
    # Gán màu đỏ cho các điểm biên
    edge_indices = canny_mask > 0
    rgb_base[edge_indices] = color_rgb
    
    return rgb_base


def crop_roi(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """
    Cắt vùng ROI kích thước size x size bắt đầu từ tọa độ (x, y) trên ảnh 512x512.
    """
    h, w = img.shape[:2]
    # Ràng buộc an toàn
    size = max(32, min(size, min(h, w)))
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))
    return img[y:y + size, x:x + size]


def draw_roi_box(img_gray: np.ndarray, x: int, y: int, size: int, color_rgb=(255, 59, 48), thickness=2) -> np.ndarray:
    """
    Vẽ khung đỏ ROI trên nền ảnh gốc (trả về ảnh RGB).
    """
    h, w = img_gray.shape[:2]
    size = max(32, min(size, min(h, w)))
    x = max(0, min(x, w - size))
    y = max(0, min(y, h - size))

    annotated = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(annotated, (x, y), (x + size, y + size), color_rgb, thickness)
    
    # Vẽ góc tiêu cự góc ngắm phong cách PACS y tế
    marker_len = 12
    # Góc trên trái
    cv2.line(annotated, (x, y), (x + marker_len, y), (0, 240, 255), 2)
    cv2.line(annotated, (x, y), (x, y + marker_len), (0, 240, 255), 2)
    # Góc dưới phải
    cv2.line(annotated, (x + size, y + size), (x + size - marker_len, y + size), (0, 240, 255), 2)
    cv2.line(annotated, (x + size, y + size), (x + size, y + size - marker_len), (0, 240, 255), 2)
    
    # Nhãn ROI
    label = f"ROI [{x},{y}:{size}]"
    cv2.putText(annotated, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

    return annotated


def process_xray_pipeline(
    image_bytes: bytes,
    filter_type: str = 'gaussian',
    kernel_size: int = 3,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    unsharp_amount: float = 1.0,
    canny_low: int = 50,
    canny_high: int = 150,
    roi_x: int = 200,
    roi_y: int = 200,
    roi_size: int = 128
) -> dict:
    """
    Toàn bộ quy trình xử lý ảnh X-quang:
    1. Đọc & chuẩn hóa Min-Max
    2. Chạy P0 (Global HE)
    3. Chạy P1 (CLAHE)
    4. Chạy P2 Pipeline: Lọc -> CLAHE -> Unsharp Masking
    5. Phát hiện biên Sobel, Canny, Overlay
    6. Cắt Zoom ROI
    7. Tính toán định lượng & đo đạc thời gian
    """
    # 0. Đọc & chuẩn hóa ảnh gốc
    t0_prep = time.perf_counter()
    img_orig = read_and_normalize_image(image_bytes, target_size=512)
    prep_time = (time.perf_counter() - t0_prep) * 1000.0

    # 1. Phương án P0: Global Histogram Equalization
    t0_p0 = time.perf_counter()
    img_p0 = cv2.equalizeHist(img_orig)
    time_p0 = (time.perf_counter() - t0_p0) * 1000.0

    # 2. Phương án P1: CLAHE đơn thuần
    t0_p1 = time.perf_counter()
    tile_size = max(2, min(tile_size, 32))
    clip_limit = max(0.1, float(clip_limit))
    clahe_p1 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    img_p1 = clahe_p1.apply(img_orig)
    time_p1 = (time.perf_counter() - t0_p1) * 1000.0

    # 3. Phương án P2: Proposed Pipeline
    t0_p2 = time.perf_counter()
    
    # Bước 3.1: Lọc nhiễu
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1  # Đảm bảo kernel lẻ
    
    if k > 1:
        if filter_type.lower() == 'median':
            step1_denoise = cv2.medianBlur(img_orig, k)
        else:
            # Gaussian
            step1_denoise = cv2.GaussianBlur(img_orig, (k, k), 0)
    else:
        step1_denoise = img_orig.copy()

    # Bước 3.2: Tăng tương phản cục bộ CLAHE
    clahe_p2 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    step2_clahe = clahe_p2.apply(step1_denoise)

    # Bước 3.3: Làm sắc nét vi cấu trúc bè xương bằng Unsharp Masking
    amount = max(0.0, float(unsharp_amount))
    if amount > 0.0:
        blurred_clahe = cv2.GaussianBlur(step2_clahe, (3, 3), 1.0)
        sharpened_float = cv2.addWeighted(step2_clahe, 1.0 + amount, blurred_clahe, -amount, 0)
        img_p2 = np.clip(sharpened_float, 0, 255).astype(np.uint8)
    else:
        img_p2 = step2_clahe.copy()

    time_p2 = (time.perf_counter() - t0_p2) * 1000.0

    # 4. Biên và Overlay
    orig_sobel, orig_canny = compute_sobel_and_canny(img_orig, canny_low, canny_high)
    p2_sobel, p2_canny = compute_sobel_and_canny(img_p2, canny_low, canny_high)

    orig_overlay = create_red_overlay(img_orig, orig_canny)
    p2_overlay = create_red_overlay(img_p2, p2_canny)

    # 5. ROI
    annotated_orig = draw_roi_box(img_orig, roi_x, roi_y, roi_size)
    roi_orig = crop_roi(img_orig, roi_x, roi_y, roi_size)
    roi_p0 = crop_roi(img_p0, roi_x, roi_y, roi_size)
    roi_p1 = crop_roi(img_p1, roi_x, roi_y, roi_size)
    roi_p2 = crop_roi(img_p2, roi_x, roi_y, roi_size)

    # 6. Tính chỉ số định lượng
    metrics_orig = compute_metrics(img_orig, img_orig, prep_time, is_reference=True)
    metrics_p0 = compute_metrics(img_orig, img_p0, time_p0, is_reference=False)
    metrics_p1 = compute_metrics(img_orig, img_p1, time_p1, is_reference=False)
    metrics_p2 = compute_metrics(img_orig, img_p2, time_p2, is_reference=False)

    # 7. Đóng gói kết quả Base64
    result = {
        "images": {
            # Tab 1: So sánh toàn cảnh & ROI
            "orig_full": encode_image_to_base64(annotated_orig, is_rgb=True),
            "orig_raw": encode_image_to_base64(img_orig),
            "p0_full": encode_image_to_base64(img_p0),
            "p1_full": encode_image_to_base64(img_p1),
            "p2_full": encode_image_to_base64(img_p2),
            "roi_orig": encode_image_to_base64(roi_orig),
            "roi_p0": encode_image_to_base64(roi_p0),
            "roi_p1": encode_image_to_base64(roi_p1),
            "roi_p2": encode_image_to_base64(roi_p2),

            # Tab 2: Các bước giải phẫu Pipeline P2
            "p2_step0": encode_image_to_base64(img_orig),
            "p2_step1": encode_image_to_base64(step1_denoise),
            "p2_step2": encode_image_to_base64(step2_clahe),
            "p2_step3": encode_image_to_base64(img_p2),

            # Tab 3: Biên & Overlay
            "orig_sobel": encode_image_to_base64(orig_sobel),
            "orig_canny": encode_image_to_base64(orig_canny),
            "orig_overlay": encode_image_to_base64(orig_overlay, is_rgb=True),
            "p2_sobel": encode_image_to_base64(p2_sobel),
            "p2_canny": encode_image_to_base64(p2_canny),
            "p2_overlay": encode_image_to_base64(p2_overlay, is_rgb=True),
        },
        "metrics": {
            "orig": metrics_orig,
            "p0": metrics_p0,
            "p1": metrics_p1,
            "p2": metrics_p2
        },
        "roi_box": {
            "x": roi_x,
            "y": roi_y,
            "size": roi_size
        },
        "params": {
            "filter_type": filter_type,
            "kernel_size": k,
            "clip_limit": clip_limit,
            "tile_size": tile_size,
            "unsharp_amount": amount,
            "canny_low": canny_low,
            "canny_high": canny_high
        }
    }

    return result
