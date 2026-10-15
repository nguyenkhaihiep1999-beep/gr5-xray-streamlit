import hashlib
import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app_utils import clamp_roi, decode_image, metrics_csv, roi_from_click, validate_image
from processor import process_xray_pipeline

ROOT = Path(__file__).resolve().parents[1]


class HelperTests(unittest.TestCase):
    def test_invalid_upload(self):
        for data in [b"", b"not an image", b"a" * (16 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                validate_image(data)

    def test_roi_boundaries_and_scaled_click(self):
        self.assertEqual(clamp_roi(0, 0, 128), (0, 0, 128))
        self.assertEqual(clamp_roi(511, -20, 128), (384, 0, 128))
        self.assertEqual(roi_from_click({"x": 128, "y": 128, "width": 256, "height": 256}, 128), (192, 192, 128))

    def test_normal_image_and_pipeline(self):
        data = (ROOT / "static/samples/sample3_knee.jpg").read_bytes()
        width, height = validate_image(data)
        self.assertGreater(width * height, 0)
        result = process_xray_pipeline(data, unsharp_amount=0.0, roi_x=0, roi_y=0)
        self.assertEqual(result["roi_box"]["x"], 0)
        self.assertEqual(result["params"]["unsharp_amount"], 0)
        self.assertEqual(len(result["images"]), 19)
        for metric in result["metrics"].values():
            self.assertEqual(sum(metric["hist"]), 512 * 512)
        image = Image.open(io.BytesIO(decode_image(result["images"]["p2_full"])))
        self.assertEqual(image.size, (512, 512))
        np.testing.assert_array_equal(
            np.array(image), np.array(Image.open(io.BytesIO(decode_image(result["images"]["p2_step2"]))))
        )
        self.assertTrue(metrics_csv(result["metrics"]).startswith(b"\xef\xbb\xbf"))

    def test_16bit_image(self):
        output = io.BytesIO()
        Image.fromarray(np.arange(1024, dtype=np.uint16).reshape(32, 32)).save(output, format="PNG")
        self.assertEqual(validate_image(output.getvalue()), (32, 32))
        result = process_xray_pipeline(output.getvalue())
        self.assertEqual(result["metrics"]["orig"]["ssim"], 1)

    def test_protected_originals(self):
        # The notebook is deliberately not part of the deployment repository.
        notebook = ROOT / "Processing/xquang_xla_skeleton .ipynb"
        if notebook.exists():
            self.assertEqual(hashlib.sha256(notebook.read_bytes()).hexdigest(),
                             "02f9abdeccdd222c31d0f3f72ac6df9370bae63a34633627891557ec975596bf")
        self.assertEqual(hashlib.sha256((ROOT / "processor.py").read_bytes()).hexdigest(),
                         "7e666e83fe761ca7d2f08921342c57e2b2978910ecdf151874e47fc96ea20b83")


class StreamlitTests(unittest.TestCase):
    def test_sample_and_controls(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        self.assertEqual(len(app.dataframe), 1)
        app.slider(key="unsharp_amount").set_value(0.0).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.result["params"]["unsharp_amount"], 0)
        app.number_input(key="roi_x").set_value(0).run()
        self.assertEqual(app.session_state.result["roi_box"]["x"], 0)
        app.selectbox[0].set_value("Đầu gối").run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
