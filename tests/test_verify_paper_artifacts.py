import math
import unittest

from verify_paper_artifacts import metrics_from_rows, summarize_decision_sources


class MetricsTest(unittest.TestCase):
    def test_metrics_are_split_by_gold_label(self):
        rows = [
            {"gold_label": "yes", "final_label": "yes"},
            {"gold_label": "yes", "final_label": "no"},
            {"gold_label": "no", "final_label": "no"},
            {"gold_label": "no", "final_label": "uncertain"},
        ]

        metrics = metrics_from_rows(rows)

        self.assertEqual(metrics["n"], 4)
        self.assertAlmostEqual(metrics["acc_eq"], 50.0)
        self.assertAlmostEqual(metrics["acc_neq"], 50.0)
        self.assertAlmostEqual(metrics["gm"], 50.0)
        self.assertAlmostEqual(metrics["overall"], 50.0)
        self.assertEqual(metrics["und"], 1)

    def test_zero_class_accuracy_yields_zero_gm(self):
        rows = [
            {"gold_label": "yes", "final_label": "no"},
            {"gold_label": "no", "final_label": "no"},
        ]

        metrics = metrics_from_rows(rows)

        self.assertEqual(metrics["acc_eq"], 0.0)
        self.assertTrue(math.isclose(metrics["gm"], 0.0))

    def test_decision_source_summary_flags_trace_prior(self):
        rows = [
            {"decision_source": "verified"},
            {"decision_source": "model_assisted"},
            {"decision_source": "trace_prior"},
        ]

        summary = summarize_decision_sources(rows)

        self.assertEqual(summary["verified"], 1)
        self.assertEqual(summary["model_assisted"], 1)
        self.assertEqual(summary["trace_prior"], 1)
        self.assertEqual(summary["non_llm_prior"], 1)


if __name__ == "__main__":
    unittest.main()
