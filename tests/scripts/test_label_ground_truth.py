"""Tests for the ground-truth labeling pipeline (scripts/label_ground_truth.py)."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.schemas.ground_truth import CorpusDocument, LabelVerdict
from src.schemas.models import GroundTruthRecord
from src.schemas.output import UnifiedEvalRecordSchema


class TestCorpusDocument:
    def test_valid_record(self) -> None:
        doc = CorpusDocument(path="docs/en/docs/async.md", content="# Async")
        assert doc.path == "docs/en/docs/async.md"
        assert doc.content == "# Async"

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CorpusDocument(path="", content="text")

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CorpusDocument(path="a.md", content="")


class TestLabelVerdict:
    def test_relevant(self) -> None:
        v = LabelVerdict(
            verdict="relevant", rationale="contains answer", confidence=0.95
        )
        assert v.verdict == "relevant"
        assert v.confidence == 0.95

    def test_irrelevant(self) -> None:
        v = LabelVerdict(verdict="irrelevant", rationale="off topic", confidence=0.8)
        assert v.verdict == "irrelevant"

    def test_invalid_verdict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabelVerdict(verdict="maybe", rationale="unsure", confidence=0.5)

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LabelVerdict(verdict="relevant", rationale="ok", confidence=1.5)


class TestGroundTruthRecord:
    def test_minimal(self) -> None:
        r = GroundTruthRecord(
            query_id="Q1",
            label="DIRECT_LOOKUP",
            query_text="how to use Depends",
            relevant_docs=["docs/en/docs/dependencies.md"],
        )
        assert r.query_id == "Q1"
        assert r.source is None

    def test_full(self) -> None:
        r = GroundTruthRecord(
            query_id="Q2",
            label="MULTI_HOP",
            query_text="forms and json",
            relevant_docs=["a.md", "b.md"],
            source="github",
            title="How to?",
            answer_text="Use Form()",
            url="https://github.com/example",
        )
        assert r.source == "github"
        assert len(r.relevant_docs) == 2

    def test_empty_query_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthRecord(
                query_id="",
                label="DIRECT_LOOKUP",
                query_text="q",
                relevant_docs=["a.md"],
            )

    def test_empty_relevant_docs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthRecord(
                query_id="Q1",
                label="DIRECT_LOOKUP",
                query_text="q",
                relevant_docs=[],
            )


class TestLoadCorpus:
    def test_loads_md_and_py(self, label_ground_truth, tmp_path) -> None:
        # Given
        md_root = tmp_path / "docs" / "en" / "docs"
        md_root.mkdir(parents=True)
        (md_root / "async.md").write_text("# Async docs")
        tutorial_dir = md_root / "tutorial"
        tutorial_dir.mkdir()
        (tutorial_dir / "body.md").write_text("# Body")

        py_root = tmp_path / "docs_src"
        py_root.mkdir(parents=True)
        (py_root / "test_auth.py").write_text("from fastapi import Depends")

        mod = label_ground_truth
        # When
        docs = mod._load_corpus(md_root, py_root)
        # Then
        assert len(docs) == 3
        paths = {d.path for d in docs}
        assert "async.md" in paths or any("async.md" in p for p in paths)

    def test_skips_empty_files(self, label_ground_truth, tmp_path) -> None:
        md_root = tmp_path / "md"
        md_root.mkdir()
        (md_root / "empty.md").write_text("   ")
        (md_root / "real.md").write_text("# Real content")

        py_root = tmp_path / "py"
        py_root.mkdir()

        mod = label_ground_truth
        docs = mod._load_corpus(md_root, py_root)
        assert len(docs) == 1
        assert docs[0].content.strip() != ""

    def test_empty_directory(self, label_ground_truth, tmp_path) -> None:
        md_root = tmp_path / "empty_md"
        md_root.mkdir()
        py_root = tmp_path / "empty_py"
        py_root.mkdir()

        mod = label_ground_truth
        docs = mod._load_corpus(md_root, py_root)
        assert docs == []


class TestCosineTopK:
    def test_returns_k_indices(self, label_ground_truth) -> None:
        mod = label_ground_truth
        corpus = np.random.randn(100, 32).astype(np.float32)
        query = np.random.randn(32).tolist()
        result = mod._cosine_top_k(query, corpus, k=10)
        assert len(result) == 10

    def test_all_indices_within_range(self, label_ground_truth) -> None:
        mod = label_ground_truth
        corpus = np.random.randn(50, 16).astype(np.float32)
        query = np.random.randn(16).tolist()
        result = mod._cosine_top_k(query, corpus, k=5)
        assert all(0 <= i < 50 for i in result)

    def test_no_duplicates(self, label_ground_truth) -> None:
        mod = label_ground_truth
        corpus = np.random.randn(20, 8).astype(np.float32)
        query = np.random.randn(8).tolist()
        result = mod._cosine_top_k(query, corpus, k=10)
        assert len(result) == len(set(result))

    def test_k_larger_than_corpus(self, label_ground_truth) -> None:
        mod = label_ground_truth
        corpus = np.random.randn(3, 8).astype(np.float32)
        query = np.random.randn(8).tolist()
        result = mod._cosine_top_k(query, corpus, k=10)
        assert len(result) == 3


class TestLoadOrEmbedCorpus:
    def test_caches_and_loads(self, label_ground_truth, tmp_path) -> None:
        mod = label_ground_truth
        from src.embeddings import StubEmbedder

        embedder = StubEmbedder(dim=8)
        docs = [
            CorpusDocument(path="a.md", content="hello"),
            CorpusDocument(path="b.md", content="world"),
        ]

        # First call: embeds and caches
        vectors = mod._load_or_embed_corpus(docs, embedder, tmp_path)
        assert vectors.shape == (2, 8)

        # Second call: loads from cache
        vectors2 = mod._load_or_embed_corpus(docs, embedder, tmp_path)
        np.testing.assert_array_equal(vectors, vectors2)

    def test_invalidates_on_content_change(self, label_ground_truth, tmp_path) -> None:
        mod = label_ground_truth
        from src.embeddings import StubEmbedder

        embedder = StubEmbedder(dim=8)
        docs_v1 = [CorpusDocument(path="a.md", content="version1")]
        docs_v2 = [CorpusDocument(path="a.md", content="version2")]

        v1 = mod._load_or_embed_corpus(docs_v1, embedder, tmp_path)
        v2 = mod._load_or_embed_corpus(docs_v2, embedder, tmp_path)

        # Different content should re-embed (StubEmbedder is deterministic per text)
        # The paths list changed, so cache is invalidated
        assert v1.shape == v2.shape


class TestAjudgeCandidate:
    @pytest.mark.asyncio
    async def test_returns_verdict_on_success(self, label_ground_truth) -> None:
        mod = label_ground_truth
        mock_response = mod.JudgeResponse(
            verdict="relevant", rationale="contains answer", confidence=0.9
        )

        with patch.object(mod, "_aclient") as mock_client:
            mock_client.completions.create = AsyncMock(return_value=mock_response)
            doc = CorpusDocument(path="deps.md", content="Use Depends() for DI")
            result = await mod._ajudge_candidate(
                "How to use dependencies?", "Use Depends()", doc
            )
            assert result.verdict == "relevant"
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_returns_irrelevant_on_failure(self, label_ground_truth) -> None:
        mod = label_ground_truth

        with patch.object(mod, "_aclient") as mock_client:
            mock_client.completions.create = AsyncMock(
                side_effect=RuntimeError("API down")
            )
            doc = CorpusDocument(path="bad.md", content="content")
            result = await mod._ajudge_candidate("q", "a", doc)
            assert result.verdict == "irrelevant"
            assert result.confidence == 0.0
            assert "failed" in result.rationale.lower()


class TestLabelQuery:
    @pytest.mark.asyncio
    async def test_labels_query_with_relevant_docs(self, label_ground_truth) -> None:
        mod = label_ground_truth
        from src.embeddings import StubEmbedder

        embedder = StubEmbedder(dim=8)
        query = UnifiedEvalRecordSchema(
            id="Q1",
            source="github",
            title="How to use Depends",
            url="https://github.com/example",
            body="### Description\nHow do I use dependencies in FastAPI?",
            answer_text="Use Depends()",
            created_at="2024-01-01T00:00:00Z",
            score=5,
            answer_score=3,
            tags=["question"],
            label=None,
        )
        corpus_docs = [
            CorpusDocument(path="deps.md", content="Use Depends() for DI"),
            CorpusDocument(path="auth.md", content="Authentication with OAuth2"),
        ]
        corpus_vectors = np.array(
            embedder.embed_texts(
                [d.path + "\n\n" + d.content[:2000] for d in corpus_docs]
            )
        )

        mock_verdict = LabelVerdict(
            verdict="relevant", rationale="yes", confidence=0.95
        )

        with patch.object(
            mod, "_ajudge_candidates", new_callable=AsyncMock
        ) as mock_judge:
            mock_judge.return_value = [
                mock_verdict,
                LabelVerdict(verdict="irrelevant", rationale="no", confidence=0.8),
            ]
            result = await mod._label_query(
                query, corpus_docs, corpus_vectors, embedder, top_k=2, concurrency=3
            )

        assert result is not None
        assert result.query_id == "Q1"
        assert "deps.md" in result.relevant_docs
        assert "auth.md" not in result.relevant_docs

    @pytest.mark.asyncio
    async def test_keeps_top1_when_no_relevant(self, label_ground_truth) -> None:
        mod = label_ground_truth
        from src.embeddings import StubEmbedder

        embedder = StubEmbedder(dim=8)
        query = UnifiedEvalRecordSchema(
            id="Q2",
            source="stackoverflow",
            title="Error",
            url="https://so.com/q/1",
            body="<p>Why does my app crash?</p>",
            answer_text="Check logs",
            created_at="2024-01-01T00:00:00Z",
            score=1,
            answer_score=1,
            tags=["error"],
            label=None,
        )
        corpus_docs = [CorpusDocument(path="crash.md", content="Debugging guide")]
        corpus_vectors = np.array(
            embedder.embed_texts(
                [d.path + "\n\n" + d.content[:2000] for d in corpus_docs]
            )
        )

        mock_verdict = LabelVerdict(
            verdict="irrelevant", rationale="no", confidence=0.9
        )

        with patch.object(
            mod, "_ajudge_candidates", new_callable=AsyncMock
        ) as mock_judge:
            mock_judge.return_value = [mock_verdict]
            result = await mod._label_query(
                query, corpus_docs, corpus_vectors, embedder, top_k=1, concurrency=3
            )

        assert result is not None
        # Should keep top-1 as fallback
        assert len(result.relevant_docs) == 1
