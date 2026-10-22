import unittest

from train import split_indices


class SplitIndicesTests(unittest.TestCase):
    def test_returns_disjoint_seventy_fifteen_fifteen_splits(self) -> None:
        train, validation, test = split_indices(10_000)

        self.assertEqual((len(train), len(validation), len(test)), (7000, 1500, 1500))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))
        self.assertEqual(len(set(train) | set(validation) | set(test)), 10_000)


if __name__ == "__main__":
    unittest.main()
