"""
Unit tests for TermSources dataclass.

Tests the per-document term tracking infrastructure including:
- Basic aggregation (total_count, num_documents)
- Confidence statistics (mean, median, std_dev)
- Quality flags (high_conf_doc_ratio, all_low_conf)
- Weighted scoring for canonical selection
"""

from src.core.vocabulary.term_sources import TermSources, merge_term_sources_dict


class TestTermSourcesBasics:
    """Test basic aggregation properties."""

    def test_empty_term_sources(self):
        """Empty TermSources should have zero values."""
        sources = TermSources()
        assert sources.total_count == 0
        assert sources.num_documents == 0
        assert sources.mean_confidence == 0.0
        assert sources.median_confidence == 0.0
        assert sources.weighted_score == 0.0

    def test_single_document(self):
        """Single document with 5 occurrences."""
        sources = TermSources.from_single_document("doc1", 0.95, 5)
        assert sources.total_count == 5
        assert sources.num_documents == 1
        assert sources.mean_confidence == 0.95
        assert sources.median_confidence == 0.95

    def test_multiple_documents(self):
        """Term appearing in multiple documents."""
        sources = TermSources(
            doc_ids=["doc1", "doc2", "doc3"],
            confidences=[0.95, 0.88, 0.60],
            counts_per_doc=[5, 2, 8],
        )
        assert sources.total_count == 15
        assert sources.num_documents == 3

    def test_add_document(self):
        """Test adding documents incrementally."""
        sources = TermSources()
        sources.add_document("doc1", 0.95, 5)
        sources.add_document("doc2", 0.60, 3)

        assert sources.num_documents == 2
        assert sources.total_count == 8

    def test_add_document_updates_existing(self):
        """Adding same doc_id should update count, not create duplicate."""
        sources = TermSources()
        sources.add_document("doc1", 0.95, 5)
        sources.add_document("doc1", 0.95, 3)  # Same doc, more occurrences

        assert sources.num_documents == 1
        assert sources.total_count == 8
        assert len(sources.doc_ids) == 1


class TestConfidenceStatistics:
    """Test confidence calculation methods."""

    def test_mean_confidence_weighted_by_count(self):
        """Mean should be weighted by occurrence count."""
        # 5 occurrences at 0.90, 1 occurrence at 0.60
        # Weighted mean = (5*0.90 + 1*0.60) / 6 = 5.1/6 = 0.85
        sources = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[0.90, 0.60],
            counts_per_doc=[5, 1],
        )
        assert abs(sources.mean_confidence - 0.85) < 0.001

    def test_median_confidence_robust_to_outliers(self):
        """Median should be robust to single outlier."""
        # 4 docs at high confidence, 1 at very low
        # Median should be high despite the outlier
        sources = TermSources(
            doc_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            confidences=[0.95, 0.92, 0.90, 0.88, 0.30],  # One bad scan
            counts_per_doc=[3, 2, 2, 2, 1],  # Low count for bad scan
        )
        # Expanded: [0.95,0.95,0.95, 0.92,0.92, 0.90,0.90, 0.88,0.88, 0.30]
        # 10 values, median is average of 5th and 6th = (0.92+0.90)/2 = 0.91
        assert sources.median_confidence > 0.85  # Still high despite outlier

    def test_std_dev_same_confidence(self):
        """Std dev should be 0 when all docs have same confidence."""
        sources = TermSources(
            doc_ids=["doc1", "doc2", "doc3"],
            confidences=[0.90, 0.90, 0.90],
            counts_per_doc=[5, 3, 2],
        )
        assert sources.confidence_std_dev == 0.0

    def test_std_dev_varying_confidence(self):
        """Std dev should be positive when confidences vary."""
        sources = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[1.0, 0.0],
            counts_per_doc=[1, 1],
        )
        # Mean = 0.5, variance = ((1-0.5)^2 + (0-0.5)^2) / 2 = 0.25
        # Std dev = 0.5
        assert abs(sources.confidence_std_dev - 0.5) < 0.001


class TestQualityFlags:
    """Test quality flag properties for ML features."""

    def test_high_conf_doc_ratio_all_high(self):
        """All docs above 0.80 should give ratio of 1.0."""
        sources = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[0.95, 0.85],
            counts_per_doc=[5, 3],
        )
        assert sources.high_conf_doc_ratio == 1.0

    def test_high_conf_doc_ratio_mixed(self):
        """Mix of high and low confidence docs."""
        sources = TermSources(
            doc_ids=["doc1", "doc2", "doc3", "doc4"],
            confidences=[0.95, 0.85, 0.70, 0.50],
            counts_per_doc=[1, 1, 1, 1],
        )
        # 2 out of 4 docs are > 0.80
        assert sources.high_conf_doc_ratio == 0.5

    def test_all_low_conf_true(self):
        """Flag should be True when all docs below 0.60."""
        sources = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[0.55, 0.50],
            counts_per_doc=[5, 3],
        )
        assert sources.all_low_conf is True

    def test_all_low_conf_false(self):
        """Flag should be False when any doc is >= 0.60."""
        sources = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[0.60, 0.50],  # doc1 is exactly at threshold
            counts_per_doc=[5, 3],
        )
        assert sources.all_low_conf is False

    def test_doc_diversity_ratio(self):
        """Test document diversity calculation."""
        sources = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[0.90, 0.85],
            counts_per_doc=[5, 3],
        )
        # 2 out of 5 total docs
        assert sources.doc_diversity_ratio(5) == 0.4
        # 2 out of 2 total docs
        assert sources.doc_diversity_ratio(2) == 1.0


class TestWeightedScore:
    """Test the weighted scoring for canonical selection."""

    def test_weighted_score_basic(self):
        """Basic weighted score calculation."""
        sources = TermSources(
            doc_ids=["doc1"],
            confidences=[1.0],
            counts_per_doc=[10],
        )
        # weighted_sum = 1.0 * 10 = 10
        # weighted_score = 10 ^ 1.1 ≈ 12.59
        assert abs(sources.weighted_score - (10**1.1)) < 0.01

    def test_weighted_score_confidence_matters(self):
        """Higher confidence should produce higher score."""
        high_conf = TermSources.from_single_document("doc1", 0.95, 10)
        low_conf = TermSources.from_single_document("doc2", 0.60, 10)

        # Same count, different confidence
        assert high_conf.weighted_score > low_conf.weighted_score

    def test_weighted_score_count_matters(self):
        """Higher count should produce higher score (exponent effect)."""
        low_count = TermSources.from_single_document("doc1", 0.90, 5)
        high_count = TermSources.from_single_document("doc2", 0.90, 50)

        # Ratio should be > 10 due to ^1.1 exponent
        ratio = high_count.weighted_score / low_count.weighted_score
        assert ratio > 10  # More than linear due to exponent

    def test_canonical_scenario_jenkins_vs_jenidns(self):
        """
        Real-world scenario: correct spelling vs OCR error.

        "Jenkins" appears in 2 high-confidence docs (5 + 2 times)
        "Jenidns" appears in 1 low-confidence doc (8 times)

        Jenkins should win despite lower total count.
        """
        jenkins = TermSources(
            doc_ids=["doc1", "doc2"],
            confidences=[0.95, 0.88],
            counts_per_doc=[5, 2],
        )
        jenidns = TermSources(
            doc_ids=["doc3"],
            confidences=[0.60],
            counts_per_doc=[8],
        )

        # Jenkins: (0.95*5 + 0.88*2)^1.1 = (4.75 + 1.76)^1.1 = 6.51^1.1 ≈ 7.80
        # Jenidns: (0.60*8)^1.1 = 4.8^1.1 ≈ 5.60
        # Jenkins wins!
        assert jenkins.weighted_score > jenidns.weighted_score


class TestMerging:
    """Test merging TermSources."""

    def test_merge_with_empty(self):
        """Merging with empty should return equivalent."""
        original = TermSources.from_single_document("doc1", 0.90, 5)
        merged = original.merge_with(TermSources())

        assert merged.total_count == 5
        assert merged.num_documents == 1

    def test_merge_different_docs(self):
        """Merging with different documents."""
        src1 = TermSources.from_single_document("doc1", 0.90, 5)
        src2 = TermSources.from_single_document("doc2", 0.80, 3)

        merged = src1.merge_with(src2)

        assert merged.total_count == 8
        assert merged.num_documents == 2

    def test_merge_same_doc(self):
        """Merging same doc should combine counts."""
        src1 = TermSources.from_single_document("doc1", 0.90, 5)
        src2 = TermSources.from_single_document("doc1", 0.90, 3)

        merged = src1.merge_with(src2)

        assert merged.total_count == 8
        assert merged.num_documents == 1

    def test_merge_term_sources_dict(self):
        """Test dictionary merging utility."""
        dict1 = {
            "Jenkins": TermSources.from_single_document("doc1", 0.90, 5),
            "Smith": TermSources.from_single_document("doc1", 0.90, 3),
        }
        dict2 = {
            "Jenkins": TermSources.from_single_document("doc2", 0.85, 2),
            "Jones": TermSources.from_single_document("doc2", 0.85, 4),
        }

        merged = merge_term_sources_dict(dict1, dict2)

        assert "Jenkins" in merged
        assert "Smith" in merged
        assert "Jones" in merged
        assert merged["Jenkins"].total_count == 7  # 5 + 2


class TestLegacyCompatibility:
    """Test backward compatibility methods."""

    def test_create_legacy(self):
        """Legacy creation should work for single-doc scenarios."""
        sources = TermSources.create_legacy(10, 0.85)

        assert sources.total_count == 10
        assert sources.num_documents == 1
        assert sources.mean_confidence == 0.85
        assert sources.doc_ids == ["legacy"]

    def test_to_dict(self):
        """Conversion to dict should include all computed properties."""
        sources = TermSources.from_single_document("doc1", 0.90, 5)
        d = sources.to_dict()

        assert "total_count" in d
        assert "mean_confidence" in d
        assert "median_confidence" in d
        assert "weighted_score" in d
        assert d["total_count"] == 5
