import unittest

import numpy as np
import pandas as pd

from predict import predict_frame


class RecordingClassifier:
    def __init__(self) -> None:
        self.seen_columns: list[str] | None = None

    def predict_proba(self, features: pd.DataFrame) -> list[np.ndarray]:
        self.seen_columns = list(features.columns)
        return [
            np.array([[0.2, 0.8], [0.9, 0.1]]),
            np.array([[0.6, 0.4], [0.1, 0.9]]),
            np.array([[0.4, 0.6], [0.8, 0.2]]),
            np.array([[0.3, 0.7], [0.6, 0.4]]),
            np.array([[0.9, 0.1], [0.5, 0.5]]),
        ]


class PredictFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RecordingClassifier()
        self.artifact = {
            "classifier": self.classifier,
            "feature_columns": ["toan_lop6", "ngu_van_lop6"],
            "target_columns": ["Vật lý", "Hóa học", "Sinh học", "Tin học", "Mỹ thuật"],
            "subject_count": 4,
        }

    def test_only_score_columns_are_used_and_id_is_preserved_as_output(self) -> None:
        frame = pd.DataFrame(
            {
                "ma_hoc_sinh": ["student-1", "student-2"],
                "toan_lop6": [8, 6],
                "ngu_van_lop6": [7, 9],
                "Vật lý": [0, 1],
            }
        )

        result = predict_frame(frame, self.artifact)

        self.assertEqual(
            self.classifier.seen_columns, ["toan_lop6", "ngu_van_lop6"]
        )
        self.assertEqual(result["ma_hoc_sinh"].tolist(), ["student-1", "student-2"])
        self.assertEqual(result["predicted::Vật lý"].tolist(), [1, 0])
        self.assertEqual(result["predicted::Hóa học"].tolist(), [1, 1])
        self.assertEqual(result.filter(like="predicted::").sum(axis=1).tolist(), [4, 4])
        self.assertEqual(result["predicted::Mỹ thuật"].tolist(), [0, 1])
        self.assertNotIn("Vật lý", result.columns)

    def test_rejects_invalid_scores(self) -> None:
        frame = pd.DataFrame({"toan_lop6": [11], "ngu_van_lop6": [7]})

        with self.assertRaisesRegex(ValueError, "ngoài khoảng"):
            predict_frame(frame, self.artifact)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "không có dòng"):
            predict_frame(
                pd.DataFrame(columns=self.artifact["feature_columns"]),
                self.artifact,
            )


if __name__ == "__main__":
    unittest.main()
