"""
Tests for the VocabTreeview class.

Verifies that the self-contained treeview correctly bundles its own
item-to-data mapping, row insertion, feedback display, and event callbacks.

NOTE: This file does not conform to the 200-line limit per user instruction.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import VF


@pytest.fixture
def mock_feedback_manager():
    """Create a mock feedback manager with configurable ratings."""
    fm = MagicMock()
    fm.get_rating.return_value = 0
    fm.get_rating_source.return_value = "session"
    return fm


@pytest.fixture
def make_term_data():
    """Factory for term data dicts."""

    def _make(term="TestTerm", score=75.0, is_person="No", found_by="NER"):
        return {
            VF.TERM: term,
            VF.QUALITY_SCORE: score,
            VF.IS_PERSON: is_person,
            VF.FOUND_BY: found_by,
            VF.OCCURRENCES: 5,
            VF.NUM_DOCS: 2,
        }

    return _make


class TestVocabTreeviewImport:
    """Verify the class is importable from the package."""

    def test_import_from_package(self):
        """VocabTreeview should be importable from src.ui.vocab_table."""
        from src.ui.vocab_table import VocabTreeview

        assert VocabTreeview is not None

    def test_import_strip_display_prefix(self):
        """strip_display_prefix should be importable from src.ui.vocab_table."""
        from src.ui.vocab_table import strip_display_prefix

        assert strip_display_prefix is not None


class TestStripDisplayPrefix:
    """Test the strip_display_prefix utility function."""

    def test_strips_link_emoji(self):
        """Should strip the link emoji prefix from terms."""
        from src.ui.vocab_table.vocab_treeview import strip_display_prefix

        result = strip_display_prefix("\U0001f517 TestTerm")
        assert result == "TestTerm"

    def test_no_prefix_unchanged(self):
        """Should return term unchanged if no prefix."""
        from src.ui.vocab_table.vocab_treeview import strip_display_prefix

        assert strip_display_prefix("TestTerm") == "TestTerm"

    def test_empty_string(self):
        """Should handle empty string."""
        from src.ui.vocab_table.vocab_treeview import strip_display_prefix

        assert strip_display_prefix("") == ""


class TestVocabTreeviewBuildValues:
    """Test _build_values without requiring Tk — isolated logic tests."""

    def test_build_values_basic(self, mock_feedback_manager, make_term_data):
        """_build_values should produce correct display values."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        # Test the static method by creating instance with mocked widget
        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = ""

                data = make_term_data(term="Gorny", score=65.7)
                columns = (VF.TERM, "Score")
                values = vtv._build_values(data, columns, rating=0)

                assert values[0] == "Gorny"
                assert "65.7" in str(values[1])

    def test_build_values_feedback_icons(self, mock_feedback_manager, make_term_data):
        """_build_values should show filled icons for rated terms."""
        from src.ui.vocab_table.column_config import THUMB_DOWN_EMPTY, THUMB_UP_FILLED
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = ""

                data = make_term_data()
                columns = (VF.TERM, VF.KEEP, VF.SKIP)
                values = vtv._build_values(data, columns, rating=1)

                assert values[1] == THUMB_UP_FILLED
                assert values[2] == THUMB_DOWN_EMPTY

    def test_build_values_duplicate_prefix(self, mock_feedback_manager, make_term_data):
        """_build_values should add link emoji for potential duplicates."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = ""

                data = make_term_data(term="Ryan")
                data["_potential_duplicate_of"] = "Ryan Hart"
                columns = (VF.TERM,)
                values = vtv._build_values(data, columns, rating=0)

                assert values[0].startswith("\U0001f517")


class TestVocabTreeviewBuildTags:
    """Test _build_tags tag construction logic."""

    def test_no_rating_returns_row_bg_only(self, mock_feedback_manager):
        """Unrated row should only have background tag."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = ""

                tags = vtv._build_tags("term", row_index=0, rating=0)
                assert tags == ("evenrow",)

                tags = vtv._build_tags("term", row_index=1, rating=0)
                assert tags == ("oddrow",)

    def test_positive_rating_session_tag(self, mock_feedback_manager):
        """Positive rating should produce rated_up_session tag."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        mock_feedback_manager.get_rating_source.return_value = "session"

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = ""

                tags = vtv._build_tags("term", row_index=0, rating=1)
                assert tags == ("evenrow", "rated_up_session")

    def test_negative_rating_loaded_tag(self, mock_feedback_manager):
        """Negative loaded rating should produce rated_down_loaded tag."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        mock_feedback_manager.get_rating_source.return_value = "loaded"

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = ""

                tags = vtv._build_tags("term", row_index=1, rating=-1)
                assert tags == ("oddrow", "rated_down_loaded")

    def test_filtered_prefix(self, mock_feedback_manager):
        """Filtered treeview should use filtered_ prefix on all tags."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        mock_feedback_manager.get_rating_source.return_value = "session"

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv._feedback_manager = mock_feedback_manager
                vtv._tag_prefix = "filtered_"

                tags = vtv._build_tags("term", row_index=0, rating=0)
                assert tags == ("filtered_evenrow",)

                tags = vtv._build_tags("term", row_index=1, rating=1)
                assert tags == ("filtered_oddrow", "filtered_rated_up_session")


class TestVocabTreeviewDataMapping:
    """Test item_to_data mapping operations."""

    def test_get_item_data_returns_empty_for_unknown(self):
        """get_item_data should return empty dict for unknown IDs."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv.item_to_data = {}
                assert vtv.get_item_data("I999") == {}

    def test_has_item_false_for_unknown(self):
        """has_item should return False for unknown IDs."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv.item_to_data = {}
                assert vtv.has_item("I001") is False

    def test_has_item_true_after_insert(self):
        """has_item should return True for known IDs."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv.item_to_data = {"I001": {"Term": "test"}}
                assert vtv.has_item("I001") is True

    def test_clear_empties_mapping(self):
        """clear() should empty item_to_data and delete treeview children."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                vtv = VocabTreeview.__new__(VocabTreeview)
                vtv.item_to_data = {"I001": {"Term": "test"}}
                vtv.widget = MagicMock()
                vtv.widget.get_children.return_value = ("I001",)

                vtv.clear()

                assert vtv.item_to_data == {}
                vtv.widget.delete.assert_called_once()


class TestVocabTreeviewNoIdCollision:
    """
    Test that two VocabTreeview instances maintain independent data mappings.

    This is the core invariant that prevents the Tk item-ID collision bug.
    """

    def test_independent_data_dicts(self, mock_feedback_manager, make_term_data):
        """Two instances should have completely independent item_to_data dicts."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview") as mock_tv:
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                # Both Tk treeviews generate "I001" as first item ID
                mock_tv.return_value.insert.return_value = "I001"

                parent = MagicMock()
                cols = (VF.TERM,)

                main_tv = VocabTreeview(parent, cols, "", mock_feedback_manager)
                filtered_tv = VocabTreeview(parent, cols, "filtered_", mock_feedback_manager)

                gorny_data = make_term_data(term="Gorny")
                follows_data = make_term_data(term="follows")

                main_tv.insert_row(gorny_data, 0, cols)
                filtered_tv.insert_row(follows_data, 0, cols)

                # Same ID "I001", but different data in each instance
                assert main_tv.get_item_data("I001")[VF.TERM] == "Gorny"
                assert filtered_tv.get_item_data("I001")[VF.TERM] == "follows"

                # Each instance correctly reports its own items
                assert main_tv.has_item("I001") is True
                assert filtered_tv.has_item("I001") is True

    def test_clear_one_does_not_affect_other(self, mock_feedback_manager, make_term_data):
        """Clearing one VocabTreeview should not affect the other."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview") as mock_tv:
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                mock_tv.return_value.insert.return_value = "I001"
                mock_tv.return_value.get_children.return_value = ("I001",)

                parent = MagicMock()
                cols = (VF.TERM,)

                main_tv = VocabTreeview(parent, cols, "", mock_feedback_manager)
                filtered_tv = VocabTreeview(parent, cols, "filtered_", mock_feedback_manager)

                main_tv.insert_row(make_term_data(term="A"), 0, cols)
                filtered_tv.insert_row(make_term_data(term="B"), 0, cols)

                main_tv.clear()

                assert not main_tv.has_item("I001")
                assert filtered_tv.has_item("I001")  # Still has its data


class TestVocabTreeviewCallbacks:
    """Test that event callbacks fire with the correct VocabTreeview instance."""

    def test_left_click_fires_callback_with_self(self, mock_feedback_manager):
        """Left click should call on_click_callback with (event, self)."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        callback = MagicMock()

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                parent = MagicMock()
                vtv = VocabTreeview(
                    parent,
                    (VF.TERM,),
                    "",
                    mock_feedback_manager,
                    on_click_callback=callback,
                )
                event = MagicMock()
                vtv._on_click(event)

                callback.assert_called_once_with(event, vtv)

    def test_right_click_fires_callback_with_self(self, mock_feedback_manager):
        """Right click should call on_right_click_callback with (event, self)."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        callback = MagicMock()

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                parent = MagicMock()
                vtv = VocabTreeview(
                    parent,
                    (VF.TERM,),
                    "",
                    mock_feedback_manager,
                    on_right_click_callback=callback,
                )
                event = MagicMock()
                vtv._on_right_click(event)

                callback.assert_called_once_with(event, vtv)

    def test_no_callback_does_not_crash(self, mock_feedback_manager):
        """Event without callback should be silently ignored."""
        from src.ui.vocab_table.vocab_treeview import VocabTreeview

        with patch("src.ui.vocab_table.vocab_treeview.ttk.Treeview"):
            with patch("src.ui.vocab_table.vocab_treeview.resolve_tags", return_value={}):
                parent = MagicMock()
                vtv = VocabTreeview(parent, (VF.TERM,), "", mock_feedback_manager)
                # Should not raise
                vtv._on_click(MagicMock())
                vtv._on_right_click(MagicMock())
