import json
import io
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from persist_chroma_parent_child import (  # noqa: E402
    build_persistent_index,
    chroma_dir_for_project,
    retrieve_parent_chunks,
)
import query_chroma_parent_child as query_module  # noqa: E402
from query_chroma_parent_child import format_hits  # noqa: E402
from query_chroma_parent_child import (  # noqa: E402
    expand_fused_candidates_to_parents,
    retrieve_hybrid_child_chunks,
)


class DeterministicEmbedding:
    """Tiny offline embedding double; Chroma itself remains real in these tests."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "refund" in lowered:
            return [0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0]

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def get_query_embedding(self, text: str) -> list[float]:
        return self._vector(text)


class PersistentParentChildChromaTests(unittest.TestCase):
    def test_rrf_combines_rank_positions_without_comparing_raw_scores(self) -> None:
        fuse = getattr(query_module, "reciprocal_rank_fusion", lambda *args, **kwargs: [])

        fused = fuse([["A", "B", "C"], ["C", "A", "D"]], rank_constant=60)

        self.assertEqual(
            [(item["child_id"], item["ranks"]) for item in fused],
            [("A", (1, 2)), ("C", (3, 1)), ("B", (2, None)), ("D", (None, 3))],
        )
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62)

    def test_bm25_ranks_the_matching_child_by_its_stable_chunk_id(self) -> None:
        rank = getattr(query_module, "bm25_rank_children", lambda *args, **kwargs: [])
        children = [
            {"chunk_id": "child-abstract", "text": "Abstracts are limited to 250 words."},
            {"chunk_id": "child-figures", "text": "Figures must be submitted separately."},
            {"chunk_id": "child-ethics", "text": "Authors must disclose conflicts of interest."},
        ]

        ranked = rank("abstracts 250 words", children, top_k=2)

        self.assertEqual(ranked[:1], ["child-abstract"])
        self.assertEqual(ranked, ["child-abstract"])

    def test_term_glossary_expands_chinese_query_with_english_aliases(self) -> None:
        expand = getattr(query_module, "expand_query_for_bm25", lambda query, aliases: query)

        expanded = expand("摘要有多少词", {"摘要": ["abstract", "abstracts"]})

        self.assertEqual(expanded, "摘要有多少词 abstract abstracts")

    def test_project_glossary_loads_the_real_abstract_translation(self) -> None:
        load_glossary = getattr(query_module, "load_term_glossary", lambda: {})
        expand = getattr(query_module, "expand_query_for_bm25", lambda query, aliases: query)

        expanded = expand("摘要有多少词", load_glossary())

        self.assertEqual(expanded, "摘要有多少词 abstract abstracts summary")

    def test_bm25_uses_glossary_alias_to_find_english_document(self) -> None:
        rank = getattr(query_module, "bm25_rank_children", lambda *args, **kwargs: [])
        children = [
            {"chunk_id": "child-abstract", "text": "The abstract is limited to 250 words."},
            {"chunk_id": "child-ethics", "text": "Authors must disclose conflicts of interest."},
            {"chunk_id": "child-fees", "text": "Article processing charges are paid after acceptance."},
        ]

        ranked = rank(
            "摘要有多少词", children, top_k=2,
            term_aliases={"摘要": ["abstract", "abstracts"]},
        )

        self.assertEqual(ranked, ["child-abstract"])

    def test_reranker_sorts_rrf_candidates_by_relevance_and_keeps_scores(self) -> None:
        rerank = getattr(query_module, "rerank_child_candidates", lambda *args, **kwargs: [])
        candidates = [
            {"chunk_id": "child-a", "text": "RRF first", "rrf_score": 0.04},
            {"chunk_id": "child-b", "text": "Reranker preferred", "rrf_score": 0.03},
        ]

        ranked = rerank(
            "which candidate answers?", candidates,
            base_url="https://rerank.example/v1", api_key="test-key", model="test-reranker",
            request_json=lambda url, headers, payload, timeout: {
                "output": {"results": [
                    {"index": 0, "relevance_score": 0.12},
                    {"index": 1, "relevance_score": 0.91},
                ]}
            },
        )

        self.assertEqual([item["chunk_id"] for item in ranked], ["child-b", "child-a"])
        self.assertEqual([item["rerank_score"] for item in ranked], [0.91, 0.12])

    def test_reranker_request_contains_query_and_candidate_child_texts(self) -> None:
        rerank = getattr(query_module, "rerank_child_candidates", lambda *args, **kwargs: [])
        requests: list[tuple[str, dict[str, str], dict[str, object], float]] = []
        candidates = [
            {"chunk_id": "child-a", "text": "Candidate text"},
        ]

        def capture_request(url, headers, payload, timeout):
            requests.append((url, headers, payload, timeout))
            return {"output": {"results": [{"index": 0, "relevance_score": 0.7}]}}

        rerank(
            "query", candidates, base_url="https://rerank.example/v1", api_key="test-key",
            model="test-reranker", request_json=capture_request,
        )

        self.assertEqual(requests[0][0], "https://rerank.example/api/v1/services/rerank/text-rerank/text-rerank")
        self.assertEqual(requests[0][1]["Authorization"], "Bearer test-key")
        self.assertEqual(requests[0][2], {
            "model": "test-reranker",
            "input": {"query": "query", "documents": ["Candidate text"]},
            "parameters": {
                "top_n": 1,
                "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
            },
        })

    def test_reranker_rejects_unusable_service_response(self) -> None:
        rerank = getattr(query_module, "rerank_child_candidates", lambda *args, **kwargs: None)

        with self.assertRaises(ValueError):
            rerank(
                "query", [{"chunk_id": "child-a", "text": "Candidate text"}],
                base_url="https://rerank.example/v1", api_key="test-key", model="test-reranker",
                request_json=lambda *args, **kwargs: {
                    "output": {"results": [{"index": 9, "relevance_score": 0.7}]}
                },
            )

    def test_reranker_rejects_partial_scores_instead_of_silently_using_rrf_order(self) -> None:
        rerank = getattr(query_module, "rerank_child_candidates", lambda *args, **kwargs: [])

        with self.assertRaises(ValueError):
            rerank(
                "query", [
                    {"chunk_id": "child-a", "text": "Candidate A"},
                    {"chunk_id": "child-b", "text": "Candidate B"},
                ],
                base_url="https://rerank.example/v1", api_key="test-key", model="test-reranker",
                request_json=lambda *args, **kwargs: {
                    "output": {"results": [{"index": 0, "relevance_score": 0.7}]}
                },
            )

    def test_hybrid_retrieval_returns_child_text_and_ranks_from_both_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children_path = root / "children.jsonl"
            children = [
                {
                    "chunk_id": "child-refund",
                    "text": "refund policy details and refund timeline",
                    "metadata": {"parent_chunk_id": "parent-refund"},
                },
                {
                    "chunk_id": "child-invoice",
                    "text": "invoice policy and billing details",
                    "metadata": {"parent_chunk_id": "parent-invoice"},
                },
                {
                    "chunk_id": "child-other-refund",
                    "text": "refund eligibility requirements",
                    "metadata": {"parent_chunk_id": "parent-other-refund"},
                },
            ]
            children_path.write_text(
                "".join(json.dumps(row) + "\n" for row in children), encoding="utf-8"
            )
            build_persistent_index(
                children_path,
                root / "chroma",
                DeterministicEmbedding(),
                collection_name="hybrid_children",
            )

            candidates = retrieve_hybrid_child_chunks(
                "refund policy", children, root / "chroma", DeterministicEmbedding(),
                collection_name="hybrid_children", candidate_k=3,
            )

            self.assertEqual(candidates[0]["chunk_id"], "child-refund")
            self.assertEqual(candidates[0]["dense_rank"], 1)
            self.assertEqual(candidates[0]["bm25_rank"], 1)
            self.assertEqual(candidates[0]["text"], "refund policy details and refund timeline")

    def test_expanding_rrf_candidates_returns_unique_parent_context_in_fused_order(self) -> None:
        candidates = [
            {
                "chunk_id": "child-1", "text": "matching evidence one",
                "metadata": {"parent_chunk_id": "parent-1"}, "rrf_score": 0.03,
                "dense_rank": 1, "bm25_rank": 2,
                "rerank_score": 0.93,
            },
            {
                "chunk_id": "child-2", "text": "duplicate parent evidence",
                "metadata": {"parent_chunk_id": "parent-1"}, "rrf_score": 0.02,
                "dense_rank": 2, "bm25_rank": None,
            },
            {
                "chunk_id": "child-3", "text": "matching evidence two",
                "metadata": {"parent_chunk_id": "parent-2"}, "rrf_score": 0.01,
                "dense_rank": None, "bm25_rank": 3,
            },
        ]
        parents = {
            "parent-1": {"text": "full parent one", "metadata": {"title": "First"}},
            "parent-2": {"text": "full parent two", "metadata": {"title": "Second"}},
        }

        hits = expand_fused_candidates_to_parents(candidates, parents, top_k=2)

        self.assertEqual([hit["parent_chunk_id"] for hit in hits], ["parent-1", "parent-2"])
        self.assertEqual(hits[0]["matched_child_id"], "child-1")
        self.assertEqual(hits[0]["matched_child_text"], "matching evidence one")
        self.assertEqual(hits[0]["rerank_score"], 0.93)
        self.assertEqual(hits[1]["bm25_rank"], 3)

    def test_query_output_includes_rank_source_page_and_parent_text(self) -> None:
        rendered = format_hits(
            [
                {
                    "parent_chunk_id": "parent-7",
                    "text": "Authors should submit the source files.",
                    "rrf_score": 0.031,
                    "dense_rank": 1,
                    "bm25_rank": 2,
                    "metadata": {
                        "title": "Submission Guidelines",
                        "source_file": "official/submission.pdf",
                        "page_start": 4,
                        "page_end": 5,
                    },
                    "rerank_score": 0.91,
                }
            ]
        )

        self.assertIn("1.", rendered)
        self.assertIn("official/submission.pdf", rendered)
        self.assertIn("4–5", rendered)
        self.assertIn("Authors should submit the source files.", rendered)
        self.assertIn("Reranker 0.9100", rendered)

    def test_console_output_replaces_characters_unsupported_by_active_encoding(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")

        try:
            emit = getattr(query_module, "emit_console_text", lambda text, out: print(text, file=out))
            emit("IEEE Access — 投稿 要求", stream)
            stream.flush()
            written = buffer.getvalue()
        except UnicodeEncodeError as error:
            self.fail(f"Console output should tolerate unsupported characters: {error}")
        finally:
            stream.close()

        self.assertIn(b"IEEE Access", written)

    def test_chroma_storage_uses_ascii_workspace_sibling_path(self) -> None:
        project_root = Path("C:/workspace/求职/投稿咨询助手项目")

        storage_path = chroma_dir_for_project(project_root)

        self.assertEqual(storage_path.parent, project_root.parents[1])
        self.assertTrue(all(ord(character) < 128 for character in str(storage_path)))

    def test_reopened_chroma_index_resolves_and_deduplicates_parent_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children_path = root / "children.jsonl"
            children = [
                {
                    "chunk_id": "child-1",
                    "text": "IEEE submission policy details",
                    "metadata": {"parent_chunk_id": "parent-1", "page_start": 2},
                },
                {
                    "chunk_id": "child-2",
                    "text": "More policy information",
                    "metadata": {"parent_chunk_id": "parent-1", "page_start": None},
                },
                {
                    "chunk_id": "child-3",
                    "text": "Refund policy details",
                    "metadata": {"parent_chunk_id": "parent-2", "page_start": 7},
                },
            ]
            children_path.write_text(
                "".join(json.dumps(row) + "\n" for row in children), encoding="utf-8"
            )
            parents = {
                "parent-1": {"chunk_id": "parent-1", "text": "Full submission policy"},
                "parent-2": {"chunk_id": "parent-2", "text": "Full refund policy"},
            }
            embedding = DeterministicEmbedding()

            indexed_count = build_persistent_index(
                children_path,
                root / "chroma",
                embedding,
                collection_name="test_children",
            )
            # Re-opening in the retrieval call must still expose all indexed children.
            hits = retrieve_parent_chunks(
                "submission policy",
                parents,
                root / "chroma",
                embedding,
                collection_name="test_children",
                top_k=3,
            )

            self.assertEqual(indexed_count, 3)
            self.assertEqual([hit["parent_chunk_id"] for hit in hits], ["parent-1", "parent-2"])
            self.assertEqual(hits[0]["text"], "Full submission policy")

    def test_reindexing_same_child_ids_does_not_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children_path = root / "children.jsonl"
            children_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "child-1",
                        "text": "Submission policy",
                        "metadata": {"parent_chunk_id": "parent-1"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            embedding = DeterministicEmbedding()

            build_persistent_index(
                children_path, root / "chroma", embedding, collection_name="test_children"
            )
            build_persistent_index(
                children_path, root / "chroma", embedding, collection_name="test_children"
            )
            hits = retrieve_parent_chunks(
                "submission policy",
                {"parent-1": {"chunk_id": "parent-1", "text": "Parent text"}},
                root / "chroma",
                embedding,
                collection_name="test_children",
                top_k=5,
            )

            self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
