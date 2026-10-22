import unittest

import pandas as pd

from preprocessing import DEFAULT_SOURCE, prepare_dataframe


class PrepareDataFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = pd.DataFrame(
            {
                "ma_hoc_sinh": ["student-1", "student-2"],
                "toan_lop6": [7, 8],
                "ngoai_ngu_lop9": [6, 9],
                " Vat ly ": ["Yes", "no"],
                "Âm nhạc": ["yes", "No"],
            }
        )

    def test_normalizes_labels_and_excludes_id_from_features(self) -> None:
        cleaned, features, labels, audit = prepare_dataframe(self.source)

        self.assertEqual(features, ["toan_lop6", "ngoai_ngu_lop9"])
        self.assertEqual(labels, ["Vat ly", "Âm nhạc"])
        self.assertEqual(cleaned["Vat ly"].tolist(), [1, 0])
        self.assertEqual(cleaned["Âm nhạc"].tolist(), [1, 0])
        self.assertNotIn("ma_hoc_sinh", features)
        self.assertEqual(audit["row_count"], 2)

    def test_rejects_out_of_range_scores_without_changing_source(self) -> None:
        original = self.source.copy(deep=True)
        self.source.loc[0, "toan_lop6"] = 11

        with self.assertRaisesRegex(ValueError, "ngoài \\[0, 10\\]"):
            prepare_dataframe(self.source)
        self.assertEqual(original.loc[0, "toan_lop6"], 7)

    def test_rejects_unknown_label_values(self) -> None:
        self.source.loc[0, " Vat ly "] = "maybe"

        with self.assertRaisesRegex(ValueError, "nhãn không hợp lệ"):
            prepare_dataframe(self.source)

    def test_accepts_binary_labels_in_current_source(self) -> None:
        source = self.source.copy()
        source[" Vat ly "] = [1, 0]
        source["Âm nhạc"] = ["1", "0"]

        cleaned, _, labels, _ = prepare_dataframe(source)

        self.assertEqual(cleaned[labels].to_numpy().tolist(), [[1, 1], [0, 0]])

    def test_default_source_is_hoc_sinh_csv(self) -> None:
        self.assertEqual(DEFAULT_SOURCE.name, "hoc_sinh.csv")


if __name__ == "__main__":
    unittest.main()
