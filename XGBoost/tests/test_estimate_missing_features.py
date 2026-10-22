import unittest

import pandas as pd

from estimate_missing_features import estimate_literature_scores


class EstimateLiteratureScoresTests(unittest.TestCase):
    def test_projects_two_grades_and_clamps_to_valid_range(self) -> None:
        source = pd.DataFrame(
            {
                "ngu_van_lop6": [6, 9, 4],
                "ngu_van_lop7": [8, 7, 4],
            }
        )

        result = estimate_literature_scores(source)

        self.assertEqual(result["ngu_van_lop8"].tolist(), [10.0, 5.0, 4.0])
        self.assertEqual(result["ngu_van_lop9"].tolist(), [10.0, 3.0, 4.0])
        self.assertNotIn("ngu_van_lop8", source.columns)
        self.assertNotIn("ngu_van_lop9", source.columns)

    def test_does_not_replace_columns_already_present(self) -> None:
        source = pd.DataFrame(
            {
                "ngu_van_lop6": [6],
                "ngu_van_lop7": [7],
                "ngu_van_lop8": [8],
            }
        )

        with self.assertRaisesRegex(ValueError, "không ghi đè"):
            estimate_literature_scores(source)


if __name__ == "__main__":
    unittest.main()
