import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import smarter_wsci  # noqa: E402


class SelectionTests(unittest.TestCase):
    def test_default_question_selects_expected_files_in_score_order(self):
        selected = smarter_wsci.select_context(smarter_wsci.DEFAULT_QUESTION)
        self.assertEqual(
            [path.name for path in selected],
            ["wifi_setup.txt", "password_changes.txt", "service_status.txt"],
        )

    def test_selection_limits_results(self):
        selected = smarter_wsci.select_context(
            "My password, email, VPN, printer and Wi-Fi all stopped working", limit=2
        )
        self.assertEqual(len(selected), 2)


class ValidationTests(unittest.TestCase):
    def test_parse_json_rejects_non_object(self):
        with self.assertRaisesRegex(ValueError, "non-object"):
            smarter_wsci.parse_json_object("[]", "test")

    def test_validation_rejects_extra_keys(self):
        value = {
            "diagnosis": "Cached credentials",
            "steps": ["Reconnect"],
            "escalate_if": ["The problem continues"],
            "extra": "not allowed",
        }
        with self.assertRaisesRegex(ValueError, "exactly"):
            smarter_wsci.validate_string_list_object(
                value, ("diagnosis", "steps", "escalate_if"), "answer"
            )


class StateTests(unittest.TestCase):
    def test_isolate_returns_only_answer_from_matching_category(self):
        state = {
            "diagnostic_contexts": {
                "wifi_setup": {
                    "problem": "secret raw question",
                    "answer": {"diagnosis": "cached credentials"},
                },
                "vpn": {"answer": {"diagnosis": "unrelated"}},
            },
            "report_context": {"total_cases": 99},
        }
        isolated = smarter_wsci.relevant_state_for("wifi_setup", state)
        self.assertEqual(
            isolated, {"previous_answer": {"diagnosis": "cached credentials"}}
        )
        self.assertNotIn("problem", json.dumps(isolated))
        self.assertNotIn("unrelated", json.dumps(isolated))

    def test_write_state_updates_total_and_category_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = smarter_wsci.empty_state()
            smarter_wsci.write_state(
                path,
                state,
                "wifi_setup",
                "question",
                [smarter_wsci.KNOWLEDGE_DIR / "wifi_setup.txt"],
                {"relevant_facts": [], "recommended_actions": [], "warnings": []},
                {"diagnosis": "d", "steps": ["s"], "escalate_if": ["e"]},
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["report_context"]["total_cases"], 1)
            self.assertEqual(
                saved["report_context"]["cases_by_category"]["wifi_setup"], 1
            )


class PipelineTests(unittest.TestCase):
    def test_pipeline_calls_model_twice_and_writes_state(self):
        compressed = {
            "relevant_facts": ["Windows may cache an old password."],
            "recommended_actions": ["Forget and reconnect to eduroam."],
            "warnings": ["Repeated failures can lock the account."],
        }
        answer = {
            "diagnosis": "The laptop probably has cached credentials.",
            "steps": ["Forget eduroam and reconnect with the new password."],
            "escalate_if": ["The account becomes locked."],
        }
        responses = [
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(compressed))),
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer))),
        ]
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with patch.object(smarter_wsci, "chat", side_effect=responses) as mock_chat:
                result = smarter_wsci.run_pipeline(
                    smarter_wsci.DEFAULT_QUESTION, "test-model", state_path
                )

            self.assertEqual(mock_chat.call_count, 2)
            self.assertEqual(result["category"], "wifi_setup")
            self.assertEqual(result["answer"], answer)
            self.assertTrue(state_path.is_file())


if __name__ == "__main__":
    unittest.main()
