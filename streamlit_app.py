"""GR5 Streamlit demo. Does not load, execute, or modify the research notebook."""

import hashlib
import io
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from app_utils import (
    METHODS, clamp_roi, decode_image, metric_rows, metrics_csv,
    roi_from_click, validate_image,
)
from processor import process_xray_pipeline

ROOT = Path(__file__).resolve().parent
SAMPLES = {
    "Cổ tay": "sample1_wrist.jpg",
    "Bàn tay": "sample2_hand.jpg",
    "Đầu gối": "sample3_knee.jpg",
}
# The deployment contains only approved, de-identified sample files.
SAMPLES = {label: name for label, name in SAMPLES.items() if (ROOT / 'static' / 'samples' / name).is_file()}
DEFAULTS = {
    "filter_type": "gaussian", "kernel_size": 3, "clip_limit": 2.0,
    "tile_size": 8, "unsharp_amount": 1.0, "canny_low": 50,
    "canny_high": 150, "roi_x": 200, "roi_y": 200, "roi_size": 128,
}


def reset_parameters():
    st.session_state.update(DEFAULTS)


def show_image(images, key, label):
    st.image(decode_image(images[key]), caption=label, width="stretch")


def main():
    st.set_page_config(page_title="GR5 · Tăng cường ảnh X-quang", page_icon="🔎", layout="wide")
    st.markdown("""
    <style>
    .block-container {max-width: 1480px; padding-top: 2.1rem; padding-bottom: 2rem;}
    h1 {font-size: 2rem !important; font-weight: 650 !important; letter-spacing: -.035em;}
    h3 {font-size: 1.05rem !important;}
    [data-testid="stSidebar"] {border-right: 1px solid #344253;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    .gr5 {display:inline-block; padding:.5rem .7rem; border-radius:10px;
        background:#1E5B9C; color:#fff; font-size:1.05rem; font-weight:700; letter-spacing:.04em;}
    .subtle {color:#AAB8C8; font-size:.92rem; margin:.25rem 0 1rem;}
    </style>
    """, unsafe_allow_html=True)

    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    pending = st.session_state.pop("pending_roi", None)
    if pending is not None:
        st.session_state.roi_x, st.session_state.roi_y, _ = pending
    x, y, size = clamp_roi(st.session_state.roi_x, st.session_state.roi_y, st.session_state.roi_size)
    st.session_state.update(roi_x=x, roi_y=y, roi_size=size)

    with st.sidebar:
        st.markdown('<span class="gr5">GR5</span>', unsafe_allow_html=True)
        st.subheader("Bàn làm việc")
        source_options = ["Ảnh mẫu", "Tải ảnh lên"] if SAMPLES else ["Tải ảnh lên"]
        source = st.radio("Nguồn ảnh", source_options, horizontal=True)
        uploaded = None
        sample = None
        if source == "Ảnh mẫu":
            sample = st.selectbox("Chọn vùng xương", list(SAMPLES))
        else:
            uploaded = st.file_uploader("Chọn ảnh X-quang", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"])
            st.caption("Tối đa 16 MB · 25 MP. Chỉ dùng ảnh đã loại thông tin cá nhân.")

        st.divider()
        st.markdown("**Điều chỉnh xử lý**")
        st.selectbox("Bộ lọc nhiễu", ["gaussian", "median"], key="filter_type",
                     format_func=lambda value: {"gaussian": "Gaussian", "median": "Median"}[value])
        st.select_slider("Kích thước kernel", [1, 3, 5, 7, 9], key="kernel_size")
        st.slider("CLAHE · clip limit", 0.5, 8.0, step=0.1, key="clip_limit")
        st.slider("CLAHE · lưới ô", 2, 32, key="tile_size")
        st.slider("Độ sắc nét · unsharp", 0.0, 3.0, step=0.1, key="unsharp_amount")
        with st.expander("Biên Canny & vùng ROI"):
            st.slider("Canny · ngưỡng thấp", 0, 254, key="canny_low")
            # Keep low < high without altering the processing algorithm.
            if st.session_state.canny_high <= st.session_state.canny_low:
                st.session_state.canny_high = st.session_state.canny_low + 1
            st.slider("Canny · ngưỡng cao", st.session_state.canny_low + 1, 255, key="canny_high")
            st.select_slider("Cạnh ROI (px)", [32, 64, 96, 128, 192, 256], key="roi_size")
            st.number_input("ROI · X", 0, 511, step=1, key="roi_x")
            st.number_input("ROI · Y", 0, 511, step=1, key="roi_y")
        st.button("Đặt lại tham số", on_click=reset_parameters, width="stretch")
        st.caption("Tham số cập nhật ngay khi thay đổi. Kernel 1: không lọc; unsharp 0: không làm sắc nét.")

    st.title("Tăng cường ảnh X-quang")
    st.markdown('<p class="subtle">Bone Fracture Multi-Region X-ray · Đồ án xử lý ảnh · Nhóm GR5</p>', unsafe_allow_html=True)
    st.caption("Đối chiếu ảnh gốc và ba phương án xử lý, quan sát chi tiết trên cùng một vùng ảnh.")

    if source == "Tải ảnh lên" and uploaded is None:
        st.info("Chọn một ảnh ở thanh bên để bắt đầu, hoặc chuyển sang ảnh mẫu.")
        return
    try:
        data = uploaded.getvalue() if uploaded is not None else (ROOT / "static" / "samples" / SAMPLES[sample]).read_bytes()
        width, height = validate_image(data)
        params = {key: st.session_state[key] for key in DEFAULTS}
        params["roi_x"], params["roi_y"], params["roi_size"] = clamp_roi(params["roi_x"], params["roi_y"], params["roi_size"])
        image_id = hashlib.sha256(data).hexdigest()
        signature = (image_id, tuple(params.items()))
        if st.session_state.get("result_signature") != signature:
            with st.spinner("Đang xử lý ảnh…"):
                result = process_xray_pipeline(data, **params)
            # Session-local only: uploaded images are not cached between users or written to disk.
            st.session_state.result = result
            st.session_state.result_signature = signature
        result = st.session_state.result
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    images, metrics = result["images"], result["metrics"]

    with st.container(border=True):
        a, b, c = st.columns([2, 1, 1])
        a.markdown(f"**{uploaded.name if uploaded is not None else sample}**")
        a.caption(f"{width} × {height} px → chuẩn hóa 512 × 512 · ảnh xám 8-bit")
        b.metric("Entropy · P2", f'{metrics["p2"]["entropy"]:.3f} bit')
        c.metric("SSIM so với gốc · P2", f'{metrics["p2"]["ssim"]:.3f}')

    comparison, pipeline, edges, numbers = st.tabs(["So sánh & ROI", "Các bước xử lý", "Biên & lớp phủ", "Chỉ số & histogram"])
    with comparison:
        st.caption("Bấm vào ảnh gốc để chọn tâm ROI. Với ảnh còn lại, dùng nút phóng to ở góc ảnh để xem chi tiết.")
        columns = st.columns(4)
        with columns[0]:
            st.markdown("**Ảnh gốc · chọn ROI**")
            image = Image.open(io.BytesIO(decode_image(images["orig_full"])))
            click = streamlit_image_coordinates(image, key=f"roi_{image_id}", width=512,
                                               use_column_width="always", cursor="crosshair")
            if click and click.get("unix_time") != st.session_state.get("last_roi_click"):
                st.session_state.last_roi_click = click.get("unix_time")
                st.session_state.pending_roi = roi_from_click(click, params["roi_size"])
                st.rerun()
        for column, key, title in zip(columns[1:], ["p0_full", "p1_full", "p2_full"], ["P0 · Cân bằng toàn cục", "P1 · CLAHE", "P2 · Pipeline đề xuất"]):
            with column:
                st.markdown(f"**{title}**")
                show_image(images, key, "512 × 512 px")
        st.markdown(f'**Vùng chi tiết** · X={params["roi_x"]}, Y={params["roi_y"]} · {params["roi_size"]} × {params["roi_size"]} px')
        for column, key, label in zip(st.columns(4), ["roi_orig", "roi_p0", "roi_p1", "roi_p2"], METHODS.values()):
            with column:
                show_image(images, key, label)

    with pipeline:
        st.caption("P2: Ảnh chuẩn hóa → Lọc nhiễu → CLAHE → Unsharp masking. P0 và P1 được xử lý riêng từ ảnh gốc.")
        for column, key, label in zip(st.columns(4), ["p2_step0", "p2_step1", "p2_step2", "p2_step3"],
                                      ["01 · Chuẩn hóa", "02 · Lọc nhiễu", "03 · CLAHE", "04 · Làm sắc nét"]):
            with column:
                show_image(images, key, label)

    with edges:
        st.caption("Cùng ngưỡng Canny cho cả hai ảnh. Đường đỏ chỉ biểu diễn biên, không phải dự đoán vị trí gãy xương.")
        for prefix, label in [("orig", "Ảnh gốc"), ("p2", "Sau xử lý P2")]:
            st.markdown(f"**{label}**")
            for column, suffix, title in zip(st.columns(3), ["sobel", "canny", "overlay"], ["Sobel", "Canny", "Lớp phủ biên"]):
                with column:
                    show_image(images, f"{prefix}_{suffix}", title)

    with numbers:
        frame = pd.DataFrame(metric_rows(metrics))
        # Keep reference text and infinity readable without mixed Arrow column types.
        frame["PSNR (dB)"] = frame["PSNR (dB)"].map(lambda value: f"{value:.2f}" if isinstance(value, (float, int)) else value)
        st.dataframe(frame, hide_index=True, width="stretch")
        histogram = pd.DataFrame([
            {"Mức xám": value, "Số điểm ảnh": count, "Phương án": METHODS[method]}
            for method in METHODS for value, count in enumerate(metrics[method]["hist"])
        ])
        chart = alt.Chart(histogram).mark_line(opacity=0.85).encode(
            x=alt.X("Mức xám:Q", scale=alt.Scale(domain=[0, 255])), y="Số điểm ảnh:Q",
            color=alt.Color("Phương án:N", scale=alt.Scale(range=["#B7C4D2", "#E6B66B", "#79BFA5", "#6AA7ED"])),
            tooltip=["Phương án", "Mức xám", "Số điểm ảnh"],
        ).properties(height=310)
        st.altair_chart(chart, width="stretch")
        st.caption("Các chỉ số tính trên toàn ảnh 512 × 512, không chỉ ROI. PSNR/SSIM tham chiếu ảnh gốc, không có ảnh sạch chuẩn. Thời gian là một lần chạy thuật toán trên máy chủ, không gồm tải ảnh hay giao diện; kết quả được giữ lại khi chỉ chuyển tab.")

    left, right = st.columns(2)
    with left:
        st.download_button("Tải ảnh P2 · PNG", decode_image(images["p2_full"]), "GR5_xray_P2.png", "image/png", width="stretch")
    with right:
        st.download_button("Tải bảng chỉ số · CSV", metrics_csv(metrics), "GR5_xray_metrics.csv", "text/csv", width="stretch")
    st.divider()
    st.caption("GR5 · Demo học thuật về tăng cường hiển thị ảnh. Không phân loại gãy xương, không dùng cho chẩn đoán. Ảnh tải lên được xử lý trong phiên làm việc, không chủ động lưu thành file.")


if __name__ == "__main__":
    main()
