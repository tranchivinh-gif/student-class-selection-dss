import unittest

import numpy as np

from selection import select_top_subjects


class SelectTopSubjectsTests(unittest.TestCase):
    def test_always_selects_four_highest_scores_and_resolves_ties_consistently(self) -> None:
        scores = np.array([
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.2, 0.2, 0.2, 0.2, 0.1],
        ])

        selected = select_top_subjects(scores)

        np.testing.assert_array_equal(selected, [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])

    def test_rejects_too_few_subjects(self) -> None:
        with self.assertRaisesRegex(ValueError, "ít nhất 4"):
            select_top_subjects(np.array([[0.8, 0.4, 0.1]]))


if __name__ == "__main__":
    unittest.main()
