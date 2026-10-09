"""Tests for AskNest's FAQ loading and answer matching behavior."""

import unittest

import chatbot_for_faqs


class ChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faqs = chatbot_for_faqs.load_faqs()
        cls.nlp = chatbot_for_faqs.load_nlp()
        cls.vectorizer, cls.matrix = chatbot_for_faqs.build_matcher(cls.faqs, cls.nlp)

    def get_answer(self, question):
        return chatbot_for_faqs.get_best_faq_answer(
            question,
            self.faqs,
            self.nlp,
            self.vectorizer,
            self.matrix,
        )

    def test_battery_question_returns_expected_answer(self):
        result = self.get_answer("How long does the battery last?")

        self.assertEqual(
            result["answer"],
            "Typical battery life is up to 20 hours of music playback depending on volume and usage.",
        )

    def test_pairing_question_returns_expected_answer(self):
        result = self.get_answer("How do I pair the headphones?")

        self.assertEqual(
            result["answer"],
            "Turn on pairing mode (press and hold the power button for 5 seconds), "
            "enable Bluetooth on your phone, and select the device from the Bluetooth list.",
        )

    def test_exact_reset_question_returns_expected_answer(self):
        result = self.get_answer("How do I reset the headphones?")

        self.assertEqual(
            result["answer"],
            "To reset: power off, then press and hold power + volume up for 10 seconds until the LED flashes.",
        )

    def test_unrelated_question_is_rejected(self):
        result = self.get_answer("What is the capital of France?")

        self.assertFalse(result["confident"])
        self.assertIsNone(result["matched_question"])

    def test_empty_input_is_handled(self):
        result = self.get_answer("")

        self.assertEqual(result["answer"], "Please enter a question.")
        self.assertIsNone(result["matched_question"])
        self.assertEqual(result["score"], 0.0)

    def test_faq_csv_contains_usable_rows(self):
        self.assertGreater(len(self.faqs), 0)
        self.assertTrue((self.faqs["question"].str.strip() != "").all())
        self.assertTrue((self.faqs["answer"].str.strip() != "").all())


if __name__ == "__main__":
    unittest.main()
