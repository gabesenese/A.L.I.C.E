from ai.learning.learning_engine import LearningEngine, TrainingExample


class _FakeVector:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeVector(self.values[item])
        if isinstance(item, _FakeVector):
            return _FakeVector([self.values[i] for i in item.values])
        if isinstance(item, list):
            return _FakeVector([self.values[i] for i in item])
        return self.values[item]

    def __iter__(self):
        return iter(self.values)


class _FakeMatrix:
    def __init__(self, rows):
        self.rows = [_FakeVector(r) for r in rows]

    def __getitem__(self, idx):
        return self.rows[idx]


class _FakeNP:
    @staticmethod
    def argpartition(values, kth):
        base = values.values if isinstance(values, _FakeVector) else values
        indexed = list(enumerate(base))
        indexed.sort(key=lambda item: item[1])
        return _FakeVector([idx for idx, _ in indexed])

    @staticmethod
    def argsort(values):
        base = values.values if isinstance(values, _FakeVector) else values
        return [idx for idx, _ in sorted(enumerate(base), key=lambda item: item[1])]


class _StubVectorizer:
    def __init__(self):
        self.transform_calls = 0
        self.fit_calls = 0

    def fit_transform(self, texts):
        self.fit_calls += 1
        return _FakeMatrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def transform(self, texts):
        self.transform_calls += 1
        return _FakeMatrix([[1.0, 0.0, 0.0]])


def test_similarity_lookup_uses_query_cache(monkeypatch, tmp_path):
    engine = LearningEngine(data_dir=str(tmp_path / "training"))
    stub_vectorizer = _StubVectorizer()

    engine.examples = [
        TrainingExample("alpha", "r1"),
        TrainingExample("beta", "r2"),
        TrainingExample("gamma", "r3"),
    ]
    engine.vectorizer = stub_vectorizer
    engine._needs_refit = True

    monkeypatch.setattr("ai.learning.learning_engine._ml_available", lambda: True)
    monkeypatch.setattr("ai.learning.learning_engine.np", _FakeNP())
    monkeypatch.setattr(
        "ai.learning.learning_engine.cosine_similarity",
        lambda _query, _matrix: _FakeMatrix([[0.9, 0.1, 0.2]]),
    )

    first = engine.get_similar_examples("alpha", top_k=2)
    second = engine.get_similar_examples("alpha", top_k=2)

    assert [ex.user_input for ex in first] == ["alpha"]
    assert [ex.user_input for ex in second] == ["alpha"]
    assert stub_vectorizer.transform_calls == 1


def test_similarity_cache_invalidates_after_collect(monkeypatch, tmp_path):
    engine = LearningEngine(data_dir=str(tmp_path / "training"))
    stub_vectorizer = _StubVectorizer()

    engine.examples = [
        TrainingExample("alpha", "r1"),
        TrainingExample("beta", "r2"),
        TrainingExample("gamma", "r3"),
    ]
    engine.vectorizer = stub_vectorizer
    engine._needs_refit = True

    monkeypatch.setattr("ai.learning.learning_engine._ml_available", lambda: True)
    monkeypatch.setattr("ai.learning.learning_engine.np", _FakeNP())
    monkeypatch.setattr(
        "ai.learning.learning_engine.cosine_similarity",
        lambda _query, _matrix: _FakeMatrix([[0.9, 0.1, 0.2]]),
    )

    engine.get_similar_examples("alpha", top_k=2)
    assert stub_vectorizer.transform_calls == 1

    engine.collect_interaction("delta", "r4")
    engine.vectorizer = stub_vectorizer
    engine.get_similar_examples("alpha", top_k=2)
    assert stub_vectorizer.transform_calls == 2
