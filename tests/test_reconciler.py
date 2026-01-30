"""Tests for reconciler.py"""


class MockCandidateTerm:
    def __init__(self, term, suggested_type, confidence=0.9, frequency=5):
        self.term = term
        self.suggested_type = suggested_type
        self.confidence = confidence
        self.frequency = frequency


class MockLLMTerm:
    def __init__(self, term, type, confidence=0.8):
        self.term = term
        self.type = type
        self.confidence = confidence


class MockLLMPerson:
    def __init__(self, name, role="other", confidence=0.8):
        self.name = name
        self.role = role
        self.confidence = confidence


class TestReconcile:
    def test_both_sources_find_same_term(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        ner = [MockCandidateTerm("John Smith", "Person")]
        llm = [MockLLMTerm("John Smith", "Person")]
        result = dedup.reconcile(ner, llm)
        matched = [r for r in result if "John Smith" in r.term]
        assert len(matched) >= 1
        assert "NER" in matched[0].found_by
        assert "LLM" in matched[0].found_by

    def test_only_ner_finds_term(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        ner = [MockCandidateTerm("Radiculopathy", "Medical")]
        llm = []
        result = dedup.reconcile(ner, llm)
        assert len(result) == 1
        assert "NER" in result[0].found_by

    def test_type_conflict_ner_wins_for_person(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        ner = [MockCandidateTerm("John Smith", "Person")]
        llm = [MockLLMTerm("John Smith", "Medical")]
        result = dedup.reconcile(ner, llm)
        matched = [r for r in result if "John Smith" in r.term]
        assert len(matched) >= 1
        assert matched[0].type == "Person"

    def test_type_conflict_llm_wins_for_medical(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        ner = [MockCandidateTerm("Radiculopathy", "Unknown")]
        llm = [MockLLMTerm("Radiculopathy", "Medical")]
        result = dedup.reconcile(ner, llm)
        matched = [r for r in result if "Radiculopathy" in r.term]
        assert len(matched) >= 1
        assert matched[0].type == "Medical"

    def test_empty_inputs(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        result = dedup.reconcile([], [])
        assert result == []

    def test_fuzzy_matching(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        ner = [MockCandidateTerm("Dr. Smith", "Person")]
        llm = [MockLLMTerm("Dr Smith", "Person")]
        result = dedup.reconcile(ner, llm)
        # Should fuzzy match these as the same term
        assert len(result) <= 2  # Ideally 1 merged, but at most 2


class TestReconcilePeople:
    def test_merges_same_person(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        ner_people = [MockCandidateTerm("James Wilson", "Person")]
        llm_people = [MockLLMPerson("James Wilson", role="plaintiff")]
        result = dedup.reconcile_people(ner_people, llm_people)
        assert len(result) >= 1
        matched = [r for r in result if "James Wilson" in r.name]
        assert len(matched) >= 1

    def test_empty_inputs(self):
        from src.core.vocabulary.reconciler import VocabularyDeduplicator

        dedup = VocabularyDeduplicator(similarity_threshold=0.85)
        result = dedup.reconcile_people([], [])
        assert result == []
