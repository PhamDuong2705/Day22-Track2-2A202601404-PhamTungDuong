"""Kiểm thử cục bộ cho các phần không cần API key hoặc kết nối mạng."""

import importlib
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"


class LocalLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step2 = importlib.import_module("02_prompt_hub_ab_routing")
        cls.step3 = importlib.import_module("03_ragas_evaluation")
        cls.step4 = importlib.import_module("04_guardrails_validator")

    def test_qa_dataset_has_50_items(self):
        from qa_pairs import QA_PAIRS, SAMPLE_QUESTIONS

        self.assertEqual(len(QA_PAIRS), 50)
        self.assertEqual(len(SAMPLE_QUESTIONS), 50)

    def test_ab_router_is_deterministic_and_uses_both_versions(self):
        request_ids = [f"req-{i:04d}" for i in range(50)]
        first = [self.step2.get_prompt_version(value) for value in request_ids]
        second = [self.step2.get_prompt_version(value) for value in request_ids]

        self.assertEqual(first, second)
        self.assertIn(self.step2.PROMPT_V1_NAME, first)
        self.assertIn(self.step2.PROMPT_V2_NAME, first)

    def test_chunking_and_faiss_retrieval(self):
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from utils.data_loader import build_vectorstore, load_knowledge_base, split_text

        chunks = split_text(load_knowledge_base(), chunk_size=575, chunk_overlap=50)
        self.assertGreater(len(chunks), 1)

        store = build_vectorstore(chunks, DeterministicFakeEmbedding(size=32))
        docs = store.as_retriever(search_kwargs={"k": 3}).invoke("LangSmith là gì?")
        self.assertEqual(len(docs), 3)
        self.assertTrue(all(doc.page_content for doc in docs))

    def test_ragas_dataset_mapping(self):
        rows = [{
            "question": "Câu hỏi",
            "answer": "Câu trả lời",
            "contexts": ["Ngữ cảnh"],
            "reference": "Đáp án chuẩn",
        }]
        dataset = self.step3.build_ragas_dataset(rows)

        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample.user_input, rows[0]["question"])
        self.assertEqual(sample.retrieved_contexts, rows[0]["contexts"])

    def test_pii_redaction_and_json_repair(self):
        from guardrails import Guard

        pii_guard = Guard().use(
            self.step4.PIIDetector(on_fail=self.step4.OnFailAction.FIX)
        )
        pii_result = pii_guard.validate(
            "Email alice@example.com, phone (555) 867-5309."
        )
        self.assertNotIn("alice@example.com", pii_result.validated_output)
        self.assertNotIn("555", pii_result.validated_output)

        json_guard = Guard().use(
            self.step4.JSONFormatter(on_fail=self.step4.OnFailAction.FIX)
        )
        json_result = json_guard.validate("```json\n{'ok': true,}\n```")
        self.assertEqual(json_result.validated_output, '{\n  "ok": true\n}')


if __name__ == "__main__":
    unittest.main()
