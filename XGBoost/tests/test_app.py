import unittest
from unittest.mock import patch

import pandas as pd

from app import web_app


class FlaskGradeFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = web_app.test_client()
        self.artifact = {
            "feature_columns": [
                "toan_lop6",
                "ngu_van_lop6",
                "ngu_van_lop7",
                "ngu_van_lop8",
                "ngu_van_lop9",
            ],
            "target_columns": ["Vật lý", "Hóa học", "Sinh học", "Tin học", "Mỹ thuật"],
        }

    @patch("app.load_model")
    def test_home_shows_grade_fields_and_sample_values(self, load_model) -> None:
        load_model.return_value = self.artifact

        response = self.client.get("/")
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('name="toan_lop6"', page)
        self.assertIn('name="ngu_van_lop6"', page)
        self.assertIn('name="ngu_van_lop8"', page)
        self.assertIn('name="ngu_van_lop9"', page)
        self.assertIn("Lớp 8</span>", page)
        self.assertNotIn("(ước tính)", page)
        self.assertIn("Hãy nhập điểm thực tế", page)
        self.assertIn('type="reset">Khôi phục điểm mẫu', page)
        self.assertIn("điểm mẫu được điền sẵn", page)
        self.assertIn("chỉ mang tính minh họa", page)

    @patch("app.predict_frame")
    @patch("app.load_model")
    def test_valid_input_displays_multilabel_results(
        self, load_model, predict
    ) -> None:
        load_model.return_value = self.artifact
        predict.return_value = pd.DataFrame(
            {
                "predicted::Vật lý": [1],
                "probability::Vật lý": [0.8],
                "predicted::Hóa học": [0],
                "probability::Hóa học": [0.3],
                "predicted::Sinh học": [1],
                "probability::Sinh học": [0.7],
                "predicted::Tin học": [1],
                "probability::Tin học": [0.6],
                "predicted::Mỹ thuật": [1],
                "probability::Mỹ thuật": [0.5],
            }
        )

        response = self.client.post(
            "/predict",
            data={
                "student_name": "Học sinh thử nghiệm",
                "toan_lop6": "8.5",
                "ngu_van_lop6": "6",
                "ngu_van_lop7": "7",
                "ngu_van_lop8": "8",
                "ngu_van_lop9": "9",
            },
        )
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Học sinh thử nghiệm", page)
        self.assertIn("Vật lý", page)
        self.assertIn("80.0%", page)
        self.assertIn("4 môn được đề xuất", page)
        self.assertNotIn("Hóa học</td>", page)
        submitted_frame = predict.call_args.args[0]
        self.assertEqual(
            submitted_frame.to_dict(orient="records"),
            [
                {
                    "toan_lop6": 8.5,
                    "ngu_van_lop6": 6.0,
                    "ngu_van_lop7": 7.0,
                    "ngu_van_lop8": 8.0,
                    "ngu_van_lop9": 9.0,
                }
            ],
        )

    @patch("app.load_model")
    def test_invalid_grade_is_rejected_and_form_is_preserved(self, load_model) -> None:
        load_model.return_value = self.artifact

        response = self.client.post(
            "/predict",
            data={
                "toan_lop6": "11",
                "ngu_van_lop6": "7",
                "ngu_van_lop7": "7",
                "ngu_van_lop8": "7",
                "ngu_van_lop9": "7",
            },
        )
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Điểm không hợp lệ", page)
        self.assertIn('value="11"', page)


if __name__ == "__main__":
    unittest.main()
