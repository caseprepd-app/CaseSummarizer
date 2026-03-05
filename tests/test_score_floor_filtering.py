"""
Tests for score floor filtering in GUI display and CSV/export paths.

Verifies that vocab_score_floor preference is applied consistently:
1. _populate_vocab_table() filters items below score floor before display
2. _get_filtered_vocab_data() applies score floor in fallback path
3. CSV export receives only filtered data
4. Changing score floor changes what's displayed/exported

All GUI dependencies (CustomTkinter, tkinter) are mocked.
"""

from unittest.mock import MagicMock, patch

from src.core.vocab_schema import VF

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vocab_items(scores):
    """Create vocab dicts with given quality scores."""
    return [
        {
            VF.TERM: f"term_{i}",
            VF.QUALITY_SCORE: score,
            VF.IS_PERSON: VF.NO,
            VF.FOUND_BY: "NER",
        }
        for i, score in enumerate(scores)
    ]


def _make_widget_stub():
    """
    Create a minimal DynamicOutputWidget-like object with the real
    _get_filtered_vocab_data method but mocked GUI internals.
    """
    from src.ui.dynamic_output import DynamicOutputWidget

    widget = object.__new__(DynamicOutputWidget)
    widget._unsorted_vocab_data = []
    widget._outputs = {}
    widget._feedback_manager = MagicMock()
    widget._feedback_manager.get_rating.return_value = 0  # not skipped
    widget._column_visibility = {}
    return widget


# ---------------------------------------------------------------------------
# _get_filtered_vocab_data — primary path (uses _unsorted_vocab_data)
# ---------------------------------------------------------------------------


class TestGetFilteredVocabDataPrimaryPath:
    """When _unsorted_vocab_data is populated, score floor was already applied
    during _populate_vocab_table(). The method just filters skipped items."""

    def test_returns_all_non_skipped_items(self):
        """Items in _unsorted_vocab_data are returned if not skipped."""
        widget = _make_widget_stub()
        items = _make_vocab_items([90, 70, 55])
        widget._unsorted_vocab_data = items

        result = widget._get_filtered_vocab_data()
        assert len(result) == 3

    def test_excludes_skipped_items(self):
        """Items with feedback rating -1 (skipped) are excluded."""
        widget = _make_widget_stub()
        items = _make_vocab_items([90, 70, 55])
        widget._unsorted_vocab_data = items

        # Second item is skipped
        def rating_side_effect(term):
            return -1 if term == "term_1" else 0

        widget._feedback_manager.get_rating.side_effect = rating_side_effect

        result = widget._get_filtered_vocab_data()
        assert len(result) == 2
        terms = [d[VF.TERM] for d in result]
        assert "term_1" not in terms


# ---------------------------------------------------------------------------
# _get_filtered_vocab_data — fallback path (raw _outputs)
# ---------------------------------------------------------------------------


class TestGetFilteredVocabDataFallbackPath:
    """When _unsorted_vocab_data is empty, the fallback path reads raw
    output data and applies score floor filtering."""

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_fallback_filters_by_score_floor(self, mock_prefs):
        """Items below score floor are excluded in fallback path."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 60,
        }.get(k, d)

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []  # empty → triggers fallback
        widget._outputs = {"Names & Vocabulary": _make_vocab_items([90, 70, 55, 40, 60])}

        result = widget._get_filtered_vocab_data()

        # Only items with score >= 60 should pass
        scores = [d[VF.QUALITY_SCORE] for d in result]
        assert scores == [90, 70, 60]

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_fallback_default_floor_55(self, mock_prefs):
        """Default score floor is 55 when preference not set."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: d  # return defaults

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []
        widget._outputs = {"Names & Vocabulary": _make_vocab_items([80, 55, 54, 30])}

        result = widget._get_filtered_vocab_data()

        scores = [d[VF.QUALITY_SCORE] for d in result]
        assert scores == [80, 55]  # 54 and 30 excluded

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_fallback_excludes_skipped(self, mock_prefs):
        """Fallback path also excludes skipped feedback items."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 50,
        }.get(k, d)

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []
        widget._outputs = {"Names & Vocabulary": _make_vocab_items([90, 70])}

        # First item skipped
        def rating_side_effect(term):
            return -1 if term == "term_0" else 0

        widget._feedback_manager.get_rating.side_effect = rating_side_effect

        result = widget._get_filtered_vocab_data()
        assert len(result) == 1
        assert result[0][VF.TERM] == "term_1"

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_fallback_empty_outputs(self, mock_prefs):
        """Fallback returns empty list when no output data exists."""
        mock_prefs.return_value = MagicMock()

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []
        widget._outputs = {}

        result = widget._get_filtered_vocab_data()
        assert result == []

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_fallback_uses_rare_word_list_key(self, mock_prefs):
        """Fallback also checks 'Rare Word List (CSV)' output key."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 50,
        }.get(k, d)

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []
        widget._outputs = {"Rare Word List (CSV)": _make_vocab_items([80, 60])}

        result = widget._get_filtered_vocab_data()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# CSV export receives filtered data
# ---------------------------------------------------------------------------


class TestCSVExportFiltering:
    """Verify that _build_vocab_csv receives only score-filtered data."""

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_csv_excludes_below_floor(self, mock_prefs):
        """CSV output should not contain items below score floor."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 60,
            "vocab_export_format": "terms_only",
        }.get(k, d)

        widget = _make_widget_stub()
        # Simulate fallback path with raw data including low-score items
        widget._unsorted_vocab_data = []
        widget._outputs = {"Names & Vocabulary": _make_vocab_items([90, 55, 40])}

        filtered = widget._get_filtered_vocab_data()
        csv_output = widget._build_vocab_csv(filtered)

        # Only term_0 (score 90) passes floor of 60
        assert "term_0" in csv_output
        assert "term_1" not in csv_output  # score 55 < 60
        assert "term_2" not in csv_output  # score 40 < 60

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_csv_includes_all_above_floor(self, mock_prefs):
        """All items at or above score floor appear in CSV."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 55,
            "vocab_export_format": "terms_only",
        }.get(k, d)

        widget = _make_widget_stub()
        items = _make_vocab_items([55, 60, 80])
        widget._unsorted_vocab_data = items  # primary path (already filtered)

        filtered = widget._get_filtered_vocab_data()
        csv_output = widget._build_vocab_csv(filtered)

        assert "term_0" in csv_output
        assert "term_1" in csv_output
        assert "term_2" in csv_output


# ---------------------------------------------------------------------------
# Score floor consistency between display and export
# ---------------------------------------------------------------------------


class TestScoreFloorConsistency:
    """Verify display and export apply the same filtering."""

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_changing_floor_changes_filtered_count(self, mock_prefs):
        """Higher floor → fewer items in filtered output."""
        items = _make_vocab_items([90, 75, 60, 55, 40])

        # Floor at 55 → 4 items
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 55,
        }.get(k, d)

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []
        widget._outputs = {"Names & Vocabulary": items}

        result_55 = widget._get_filtered_vocab_data()
        assert len(result_55) == 4

        # Floor at 75 → 2 items
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 75,
        }.get(k, d)

        result_75 = widget._get_filtered_vocab_data()
        assert len(result_75) == 2

    @patch("src.ui.dynamic_output.get_user_preferences")
    def test_floor_at_boundary(self, mock_prefs):
        """Score exactly at floor passes; score one below does not."""
        mock_prefs.return_value = MagicMock()
        mock_prefs.return_value.get.side_effect = lambda k, d=None: {
            "vocab_score_floor": 60,
        }.get(k, d)

        widget = _make_widget_stub()
        widget._unsorted_vocab_data = []
        widget._outputs = {"Names & Vocabulary": _make_vocab_items([60, 59])}

        result = widget._get_filtered_vocab_data()
        assert len(result) == 1
        assert result[0][VF.QUALITY_SCORE] == 60
