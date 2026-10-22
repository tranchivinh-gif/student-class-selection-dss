import unittest
from pathlib import Path
from unittest.mock import patch

from run_pipeline import resolve_source, run_pipeline


class PipelineOrderTests(unittest.TestCase):
    def test_runs_all_stages_in_order_with_same_source(self) -> None:
        calls = []
        source = Path(__file__)

        with patch("builtins.print"), \
             patch("run_pipeline.create_eda_report", side_effect=lambda path: calls.append(("eda", path))), \
             patch("run_pipeline.prepare_data", side_effect=lambda path: calls.append(("clean", path))), \
             patch("run_pipeline.train_model", side_effect=lambda: calls.append(("train", None))), \
             patch("run_pipeline.evaluate_main", side_effect=lambda: calls.append(("evaluate", None))), \
             patch("run_pipeline.analyze", side_effect=lambda: calls.append(("present", None))):
            run_pipeline(source)

        self.assertEqual(calls, [
            ("eda", source),
            ("clean", source),
            ("train", None),
            ("evaluate", None),
            ("present", None),
        ])

    def test_data_shorthand_resolves_to_project_data_directory(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "data" / "hoc_sinh.csv"
        self.assertEqual(resolve_source(Path("data/hoc_sinh.csv")), expected)


if __name__ == "__main__":
    unittest.main()
