# GR5 · Tăng cường ảnh X-quang

Demo xử lý ảnh cho **Bone Fracture Multi-Region X-ray**. Giao diện chính dùng Streamlit, nền tối dịu và xanh chủ đạo. Giữ nguyên thuật toán web trong `processor.py`; không dùng mô hình phân loại và không phục vụ chẩn đoán.

## Chạy Streamlit

Python 3.11. Tạo môi trường riêng để không thay đổi môi trường chạy notebook:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Ảnh mẫu có sẵn; có thể tải JPG/PNG/TIFF/BMP/WebP tối đa 16 MB và 25 MP. Ảnh được chuẩn hóa min-max, đổi xám 8-bit, resize vuông 512 × 512 như bản web cũ. Không hỗ trợ DICOM. Chỉ dùng ảnh đã loại bỏ thông tin cá nhân. Ảnh tải lên được giữ trong phiên, không chủ động lưu ra đĩa hoặc cache dùng chung giữa người dùng.

Ba ảnh mẫu cục bộ được giữ nguyên. Hai ảnh cổ tay/bàn tay có thông tin dạng tên, ngày sinh và mã bệnh nhân chèn lên ảnh nên bị loại khỏi Git. Bản triển khai hiện chỉ dùng ảnh đầu gối không có các thông tin này, chờ chủ dự án duyệt phương án xử lý hai ảnh còn lại. Danh sách mẫu tự nhận những file có sẵn.

Tính năng: so sánh gốc/P0 HE/P1 CLAHE/P2, chọn ROI bằng click hoặc tọa độ, quan sát từng bước P2, Sobel/Canny/lớp phủ biên, histogram và chỉ số tính trực tiếp, tải ảnh P2 và CSV UTF-8 có BOM.

## Tổ chức thư mục

```text
streamlit_app.py       Giao diện chính
app_utils.py           Kiểm tra đầu vào, ROI, xuất CSV
processor.py           Thuật toán web gốc, giữ nguyên
requirements.txt       Thư viện cho Streamlit
.streamlit/config.toml Theme và giới hạn upload
static/samples/        Ba ảnh mẫu dùng chung
tests/                 Kiểm thử tự động
legacy/flask/          Web Flask, HTML/CSS/JS, cấu hình Render cũ
scripts/               Công cụ tạo biểu đồ và xuất báo cáo Word
reports/               Hai báo cáo Word hiện có, không tạo lại
outputs/figures/       Sáu biểu đồ báo cáo hiện có
Processing/            Notebook gốc và kết quả thực nghiệm, không chỉnh sửa
data/                  Dữ liệu cục bộ, giữ nguyên
```

`Processing/`, `data/`, `reports/`, `outputs/`, môi trường ảo và secrets bị loại khỏi Git bằng `.gitignore`. Chúng **vẫn còn trên máy**, chỉ không được đưa lên kho triển khai. Không đổi tên, di chuyển, chạy hay sửa notebook trong quá trình chuyển sang Streamlit.

Kho GitHub này phục vụ triển khai demo: không bao gồm các script báo cáo hay bản Flask lưu riêng trên máy. Bố cục ở trên mô tả toàn bộ dự án cục bộ, không phải toàn bộ nội dung kho triển khai.

## Kiểm duyệt và giới hạn cần biết

- Thư mục `data/` cục bộ có 3.787 file, không phải toàn bộ dataset. Kết quả `Processing/outputs/metrics.csv` ghi 10.563 ảnh khác nhau từ lần chạy notebook trước; không được đánh đồng hai tập này.
- Một số giá trị và kết luận trong script/báo cáo là số viết sẵn. Các báo cáo được giữ nguyên, không tự xác nhận rằng số đó khớp CSV thực nghiệm. Không chạy lại các script nếu chưa đối chiếu số liệu và sao lưu báo cáo cũ.
- Web và notebook không hoàn toàn cùng cách cài đặt: chuẩn hóa 8-bit, kernel unsharp, làm mượt trước dò biên và màu overlay có khác biệt. Streamlit tái sử dụng **web hiện có**, không tự sửa notebook để ép khớp.
- PSNR/SSIM đo độ giống ảnh gốc, không phải chất lượng so với ảnh sạch chuẩn. Entropy/độ sắc nét cao hơn không mặc nhiên tốt hơn cho chẩn đoán. Chỉ số nhiễu là phần dư cao tần ước lượng.
- Thời gian đo là một lần thực thi trên máy chủ, không phải thống kê toàn dataset. Kết quả có thể khác môi trường notebook.
- Kiểm tra đầu vào được bổ sung ở giao diện Streamlit, không sửa lõi `processor.py`. Bộ lọc, Canny và ROI vẫn dùng thuật toán web gốc.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Triển khai Streamlit Community Cloud

1. Đẩy mã nguồn, `.streamlit/config.toml`, `requirements.txt` và chỉ ảnh mẫu đã kiểm tra thông tin cá nhân lên GitHub; giữ kho riêng tư mặc định.
2. Trong Streamlit Community Cloud chọn repo, branch `main`, entrypoint `streamlit_app.py`.
3. Chọn Python **3.11** trong Advanced settings rồi Deploy.
4. Kiểm tra các ảnh mẫu được phép triển khai, tham số, ROI và tải kết quả trên URL được cấp. Không tải notebook/dataset/báo cáo lên để chạy demo.

Tài liệu chính thức: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Bản Flask và báo cáo cũ

Bản Flask được giữ tại `legacy/flask/`. Tài liệu trước khi sắp xếp nằm ở `legacy/flask/README.original.md`; các đường dẫn trong tài liệu đó mô tả bố cục cũ.

```powershell
# Dùng môi trường riêng nếu muốn chạy Flask.
python -m pip install -r legacy/flask/requirements.txt
python legacy/flask/app.py
```

Các script ở `scripts/` đã được chỉnh đường dẫn theo vị trí dự án thay vì ổ `D:` cố định, không sửa nội dung báo cáo. Chúng có thể ghi đè báo cáo/hình ảnh cùng tên; chỉ chạy khi đã chủ động kiểm tra dữ liệu.
