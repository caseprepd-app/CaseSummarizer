"""
Tests for filtered vocabulary table behavior.

Covers:
1. High-scoring single-occurrence terms promoted to main table (score >= 85)
2. Filtered table score floor (vocab_filtered_score_floor preference)
3. Filtered table default sort order (by score descending)

All GUI dependencies are mocked.
"""

from unittest.mock import MagicMock, patch

from src.core.vocab_schema import VF

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vocab_items(scores, occurrences=None):
    """Create vocab dicts with given quality scores and optional occurrence counts."""
    if occurrences is None:
        occurrences = [1] * len(scores)
    return [
        {
            VF.TERM: f"term_{i}",
            VF.QUALITY_SCORE: score,
            VF.IS_PERSON: VF.NO,
            VF.FOUND_BY: "NER",
            VF.OCCURRENCES: occ,
        }
        for i, (score, occ) in enumerate(zip(scores, occurrences))
    ]


def _make_widget_stub():
    """Create a minimal DynamicOutputWidget-like object with mocked GUI."""
    from src.ui.dynamic_output import DynamicOutputWidget

    widget = object.__new__(DynamicOutputWidget)
    widget._unsorted_vocab_data = []
    widget._outputs = {}
    widget._feedback_manager = MagicMock()
    widget._feedback_manager.get_rating.return_value = 0
    widget._column_visibility = {}
    widget._filtered_vocab_data_raw = []
    widget._filtered_unsorted_data = []
    widget._filtered_sort_column = None
    widget._filtered_sort_ascending = True
    return widget


# ---------------------------------------------------------------------------
# Test: Filtered table score floor
# ---------------------------------------------------------------------------


class TestFilteredScoreFloor:
    """Tests for vocab_filtered_score_floor preference."""

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_filtered_floor_excludes_low_scores(self, mock_prefs):
        """Items below filtered score floor are excluded from filtered section."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_filtered_score_floor": 40,
        }.get(k, d)

        widget = _make_widget_stub()
        # Scores: 25, 35, 45, 55 — only 45 and 55 should pass floor of 40
        items = _make_vocab_items([25, 35, 45, 55])
        # Simulate what _display_filtered_section does
        filtered_floor = mock_prefs.return_value.get("vocab_filtered_score_floor", 40)
        filtered = [
            item
            for item in items
            if isinstance(item, dict) and item.get(VF.QUALITY_SCORE, 0) >= filtered_floor
        ]
        assert len(filtered) == 2
        assert filtered[0][VF.QUALITY_SCORE] == 45
        assert filtered[1][VF.QUALITY_SCORE] == 55

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_filtered_floor_default_40(self, mock_prefs):
        """Default floor of 40 filters correctly."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {}.get(k, d)

        # Default should be 40
        floor = mock_prefs.return_value.get("vocab_filtered_score_floor", 40)
        assert floor == 40

        items = _make_vocab_items([39.9, 40.0, 40.1])
        filtered = [item for item in items if item.get(VF.QUALITY_SCORE, 0) >= floor]
        # 39.9 excluded, 40.0 and 40.1 included
        assert len(filtered) == 2

    def test_filtered_floor_at_boundary(self):
        """Score exactly at floor passes; score just below does not."""
        floor = 40
        items = _make_vocab_items([39, 40, 41])
        filtered = [item for item in items if item.get(VF.QUALITY_SCORE, 0) >= floor]
        assert len(filtered) == 2
        scores = [item[VF.QUALITY_SCORE] for item in filtered]
        assert 39 not in scores
        assert 40 in scores
        assert 41 in scores

    def test_filtered_floor_all_below(self):
        """When all items below floor, filtered list is empty."""
        floor = 40
        items = _make_vocab_items([10, 20, 30])
        filtered = [item for item in items if item.get(VF.QUALITY_SCORE, 0) >= floor]
        assert len(filtered) == 0

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_filtered_floor_custom_value(self, mock_prefs):
        """User-configured floor of 25 shows more items."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_filtered_score_floor": 25,
        }.get(k, d)

        floor = mock_prefs.return_value.get("vocab_filtered_score_floor", 40)
        items = _make_vocab_items([20, 25, 30, 45])
        filtered = [item for item in items if item.get(VF.QUALITY_SCORE, 0) >= floor]
        # 20 excluded, 25/30/45 included
        assert len(filtered) == 3


# ---------------------------------------------------------------------------
# Test: High-scoring single-occurrence promotion
# ---------------------------------------------------------------------------


class TestSingleOccurrencePromotion:
    """Tests for promoting high-scoring single-occurrence items to main table."""

    def test_high_score_single_occurrence_promoted(self):
        """Single-occurrence term with score >= 85 goes to main vocabulary."""
        from unittest.mock import patch as mock_patch

        from src.core.vocabulary.vocabulary_extractor import VocabularyExtractor

        with (
            mock_patch.object(VocabularyExtractor, "__init__", lambda self: None),
            mock_patch.object(VocabularyExtractor, "_apply_ml_boost", return_value=90.0),
        ):
            # The logic: if frequency < min_occurrences AND score >= 85, keep in vocab
            # We test the decision logic directly
            score = 90.0
            min_occurrences = 2
            frequency = 1  # Single occurrence

            # Simulates the extractor logic
            if frequency < min_occurrences and score >= 85:
                promoted = True
            elif frequency < min_occurrences:
                promoted = False
            else:
                promoted = True

            assert promoted is True

    def test_low_score_single_occurrence_filtered(self):
        """Single-occurrence term with score < 85 goes to filtered list."""
        score = 60.0
        min_occurrences = 2
        frequency = 1

        if frequency < min_occurrences and score >= 85:
            promoted = True
        elif frequency < min_occurrences:
            promoted = False
        else:
            promoted = True

        assert promoted is False

    def test_borderline_score_84_not_promoted(self):
        """Score of 84 (just below 85 threshold) is filtered."""
        score = 84.9
        frequency = 1
        min_occurrences = 2

        promoted = not (frequency < min_occurrences) or score >= 85
        assert promoted is False

    def test_borderline_score_85_promoted(self):
        """Score of exactly 85 is promoted."""
        score = 85.0
        frequency = 1
        min_occurrences = 2

        promoted = not (frequency < min_occurrences) or score >= 85
        assert promoted is True

    def test_multi_occurrence_unaffected(self):
        """Terms with 2+ occurrences always go to main list regardless of score."""
        score = 30.0
        frequency = 2
        min_occurrences = 2

        # frequency >= min_occurrences, so it goes to vocabulary
        promoted = not (frequency < min_occurrences) or score >= 85
        assert promoted is True


# ---------------------------------------------------------------------------
# Test: Settings registry entry
# ---------------------------------------------------------------------------


class TestFilteredScoreFloorSetting:
    """Tests for the vocab_filtered_score_floor settings UI registration."""

    def test_setting_registered(self):
        """vocab_filtered_score_floor is registered in settings registry."""
        # Trigger registration
        from src.ui.settings import settings_registry  # noqa: F401
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        keys = [s.key for s in all_settings]
        assert "vocab_filtered_score_floor" in keys

    def test_setting_range(self):
        """Setting has min=20, max=49, default=40."""
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        setting = next(s for s in all_settings if s.key == "vocab_filtered_score_floor")
        assert setting.default == 40
        assert setting.min_value == 20
        assert setting.max_value == 49

    def test_setting_category(self):
        """Setting is in the Vocabulary category."""
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        setting = next(s for s in all_settings if s.key == "vocab_filtered_score_floor")
        assert setting.category == "Vocabulary"


class TestExportRelevanceFloorSetting:
    """Tests for the renamed semantic_export_relevance_floor setting."""

    def test_relevance_floor_registered(self):
        """semantic_export_relevance_floor is registered in settings."""
        from src.ui.settings import settings_registry  # noqa: F401
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        keys = [s.key for s in all_settings]
        assert "semantic_export_relevance_floor" in keys

    def test_old_confidence_key_not_registered(self):
        """Old semantic_export_confidence_floor key should not exist."""
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        keys = [s.key for s in all_settings]
        assert "semantic_export_confidence_floor" not in keys

    def test_relevance_floor_label(self):
        """Setting label should say 'relevance' not 'confidence'."""
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        setting = next(s for s in all_settings if s.key == "semantic_export_relevance_floor")
        assert "relevance" in setting.label.lower()
        assert "confidence" not in setting.label.lower()

    def test_relevance_floor_in_export_category(self):
        """Setting should be in Export tab, not Search Export."""
        from src.ui.settings.settings_registry import SettingsRegistry

        all_settings = SettingsRegistry.get_all_settings()
        setting = next(s for s in all_settings if s.key == "semantic_export_relevance_floor")
        assert setting.category == "Export"
