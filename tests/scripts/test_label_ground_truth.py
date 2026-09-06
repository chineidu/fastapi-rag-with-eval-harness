"""Tests for the ground-truth labeling pipeline (scripts/label_ground_truth.py)."""

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.embeddings import StubEmbedder
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

    def test_empty_relevant_docs_with_answerable_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthRecord(
                query_id="Q1",
                label="DIRECT_LOOKUP",
                query_text="q",
                relevant_docs=[],
            )

    def test_unanswerable_allows_empty_relevant_docs(self) -> None:
        r = GroundTruthRecord(
            query_id="Q1",
            label="DIRECT_LOOKUP",
            query_text="q",
            relevant_docs=[],
            answerable=False,
        )
        assert r.answerable is False
        assert r.relevant_docs == []

    def test_unanswerable_with_docs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GroundTruthRecord(
                query_id="Q1",
                label="DIRECT_LOOKUP",
                query_text="q",
                relevant_docs=["a.md"],
                answerable=False,
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
        # No relevant docs found -> marked unanswerable, no synthetic fallback
        assert result.answerable is False
        assert result.relevant_docs == []


def _make_query(qid: str) -> UnifiedEvalRecordSchema:
    """Build a minimal UnifiedEvalRecordSchema for resume tests."""
    return UnifiedEvalRecordSchema(
        id=qid,
        source="github",
        title=f"Question {qid}",
        url=f"https://example.com/{qid}",
        body=f"body {qid}",
        answer_text="ans",
        created_at="2024-01-01T00:00:00Z",
        score=1,
        answer_score=1,
        tags=["q"],
        label=None,
    )


def _make_gt_record(qid: str) -> GroundTruthRecord:
    """Build a minimal GroundTruthRecord for resume fixtures."""
    return GroundTruthRecord(
        query_id=qid,
        label="DIRECT_LOOKUP",
        query_text=f"body {qid}",
        relevant_docs=["a.md"],
    )


class TestAlabelAll:
    @pytest.mark.asyncio
    async def test_resume_skips_done_ids(self, label_ground_truth, tmp_path) -> None:
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        out_path.write_text(
            json.dumps(_make_gt_record("Q0").model_dump()) + "\n",
            encoding="utf-8",
        )
        queries = [_make_query(qid) for qid in ("Q0", "Q1", "Q2")]
        embedder = StubEmbedder(dim=8)
        corpus_docs = [CorpusDocument(path="a.md", content="x")]
        corpus_vectors = np.array([[0.0] * 8], dtype=np.float32)

        processed: list[str] = []

        async def fake_label_query(
            query: UnifiedEvalRecordSchema, *args: object, **kwargs: object
        ) -> GroundTruthRecord:
            processed.append(query.id)
            return _make_gt_record(query.id)

        # When
        with patch.object(mod, "_label_query", side_effect=fake_label_query):
            results = await mod._alabel_all(
                queries,
                corpus_docs,
                corpus_vectors,
                embedder,
                top_k=1,
                concurrency=2,
                dry_run=False,
                out_path=out_path,
            )

        # Then
        assert processed == ["Q1", "Q2"]
        assert [r.query_id for r in results] == ["Q1", "Q2"]
        lines = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        assert len(lines) == 3
        assert {json.loads(ln)["query_id"] for ln in lines} == {"Q0", "Q1", "Q2"}

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_done(
        self, label_ground_truth, tmp_path
    ) -> None:
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        out_path.write_text(
            json.dumps(_make_gt_record("Q0").model_dump()) + "\n",
            encoding="utf-8",
        )
        embedder = StubEmbedder(dim=8)

        # When
        results = await mod._alabel_all(
            [_make_query("Q0")],
            [CorpusDocument(path="a.md", content="x")],
            np.array([[0.0] * 8], dtype=np.float32),
            embedder,
            top_k=1,
            concurrency=1,
            dry_run=False,
            out_path=out_path,
        )

        # Then
        assert results == []

    @pytest.mark.asyncio
    async def test_truncates_legacy_json_array(
        self, label_ground_truth, tmp_path
    ) -> None:
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        legacy = json.dumps([_make_gt_record("Q0").model_dump()])
        out_path.write_text(legacy, encoding="utf-8")
        embedder = StubEmbedder(dim=8)

        async def fake_label_query(
            query: UnifiedEvalRecordSchema, *args: object, **kwargs: object
        ) -> GroundTruthRecord:
            return _make_gt_record(query.id)

        # When
        with patch.object(mod, "_label_query", side_effect=fake_label_query):
            results = await mod._alabel_all(
                [_make_query("Q1")],
                [CorpusDocument(path="a.md", content="x")],
                np.array([[0.0] * 8], dtype=np.float32),
                embedder,
                top_k=1,
                concurrency=1,
                dry_run=False,
                out_path=out_path,
            )

        # Then
        assert [r.query_id for r in results] == ["Q1"]
        content = out_path.read_text(encoding="utf-8").strip()
        assert not content.startswith("[")
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["query_id"] == "Q1"

    @pytest.mark.asyncio
    async def test_skips_malformed_lines(self, label_ground_truth, tmp_path) -> None:
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        valid = json.dumps(_make_gt_record("Q0").model_dump())
        out_path.write_text(f"{valid}\nthis is not valid json\n", encoding="utf-8")
        embedder = StubEmbedder(dim=8)

        processed: list[str] = []

        async def fake_label_query(
            query: UnifiedEvalRecordSchema, *args: object, **kwargs: object
        ) -> GroundTruthRecord:
            processed.append(query.id)
            return _make_gt_record(query.id)

        # When
        with patch.object(mod, "_label_query", side_effect=fake_label_query):
            await mod._alabel_all(
                [_make_query("Q0"), _make_query("Q1")],
                [CorpusDocument(path="a.md", content="x")],
                np.array([[0.0] * 8], dtype=np.float32),
                embedder,
                top_k=1,
                concurrency=1,
                dry_run=False,
                out_path=out_path,
            )

        # Then
        assert processed == ["Q1"]

    @pytest.mark.asyncio
    async def test_dry_run_emits_placeholder_without_judge(
        self, label_ground_truth, tmp_path
    ) -> None:
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        embedder = StubEmbedder(dim=8)

        # When
        with patch.object(mod, "_label_query") as mock_label:
            results = await mod._alabel_all(
                [_make_query("Q1")],
                [CorpusDocument(path="a.md", content="x")],
                np.array([[0.0] * 8], dtype=np.float32),
                embedder,
                top_k=1,
                concurrency=1,
                dry_run=True,
                out_path=out_path,
            )

        # Then
        mock_label.assert_not_called()
        assert len(results) == 1
        assert results[0].query_id == "Q1"
        assert results[0].relevant_docs == ["(dry-run)"]
        lines = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        assert len(lines) == 1
        assert json.loads(lines[0])["relevant_docs"] == ["(dry-run)"]

    @pytest.mark.asyncio
    async def test_incremental_writes_one_record_per_query(
        self, label_ground_truth, tmp_path
    ) -> None:
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        embedder = StubEmbedder(dim=8)
        queries = [_make_query(f"Q{i}") for i in range(3)]

        async def fake_label_query(
            query: UnifiedEvalRecordSchema, *args: object, **kwargs: object
        ) -> GroundTruthRecord:
            return _make_gt_record(query.id)

        # When
        with patch.object(mod, "_label_query", side_effect=fake_label_query):
            results = await mod._alabel_all(
                queries,
                [CorpusDocument(path="a.md", content="x")],
                np.array([[0.0] * 8], dtype=np.float32),
                embedder,
                top_k=1,
                concurrency=1,
                dry_run=False,
                out_path=out_path,
            )

        # Then
        assert [r.query_id for r in results] == ["Q0", "Q1", "Q2"]
        lines = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        assert len(lines) == 3
        parsed = [json.loads(ln) for ln in lines]
        assert [p["query_id"] for p in parsed] == ["Q0", "Q1", "Q2"]

    @pytest.mark.asyncio
    async def test_limit_stops_after_n_remaining(
        self, label_ground_truth, tmp_path
    ) -> None:
        """Given a limit, then only the first N remaining queries are processed."""
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        embedder = StubEmbedder(dim=8)
        queries = [_make_query(f"Q{i}") for i in range(3)]
        processed: list[str] = []

        async def fake_label_query(
            query: UnifiedEvalRecordSchema, *args: object, **kwargs: object
        ) -> GroundTruthRecord:
            processed.append(query.id)
            return _make_gt_record(query.id)

        # When
        with patch.object(mod, "_label_query", side_effect=fake_label_query):
            results = await mod._alabel_all(
                queries,
                [CorpusDocument(path="a.md", content="x")],
                np.array([[0.0] * 8], dtype=np.float32),
                embedder,
                top_k=1,
                concurrency=1,
                dry_run=False,
                out_path=out_path,
                limit=2,
            )

        # Then
        assert processed == ["Q0", "Q1"]
        assert [r.query_id for r in results] == ["Q0", "Q1"]
        lines = [
            ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        assert len(lines) == 2
        assert {json.loads(ln)["query_id"] for ln in lines} == {"Q0", "Q1"}

    @pytest.mark.asyncio
    async def test_limit_applies_to_remaining_after_resume(
        self, label_ground_truth, tmp_path
    ) -> None:
        """Given a resume with a limit, then the limit counts only remaining queries."""
        # Given
        mod = label_ground_truth
        out_path = tmp_path / "gt.jsonl"
        out_path.write_text(
            json.dumps(_make_gt_record("Q0").model_dump()) + "\n",
            encoding="utf-8",
        )
        embedder = StubEmbedder(dim=8)
        queries = [_make_query(qid) for qid in ("Q0", "Q1", "Q2", "Q3")]
        processed: list[str] = []

        async def fake_label_query(
            query: UnifiedEvalRecordSchema, *args: object, **kwargs: object
        ) -> GroundTruthRecord:
            processed.append(query.id)
            return _make_gt_record(query.id)

        # When
        with patch.object(mod, "_label_query", side_effect=fake_label_query):
            results = await mod._alabel_all(
                queries,
                [CorpusDocument(path="a.md", content="x")],
                np.array([[0.0] * 8], dtype=np.float32),
                embedder,
                top_k=1,
                concurrency=1,
                dry_run=False,
                out_path=out_path,
                limit=2,
            )

        # Then
        # Q0 skipped (already done); limit 2 covers Q1 and Q2 only.
        assert processed == ["Q1", "Q2"]
        assert [r.query_id for r in results] == ["Q1", "Q2"]
