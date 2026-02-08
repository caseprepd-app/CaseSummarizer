"""
Tests for the TextFindBar widget (Ctrl+F find-in-text).

Tests search, highlighting, navigation, show/hide, and edge cases.
"""

import pytest

# Skip all tests if display is not available (CI environment)
pytest.importorskip("tkinter")


@pytest.fixture
def tk_root():
    """Create a hidden Tk root window for testing."""
    import customtkinter as ctk

    root = ctk.CTk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def container(tk_root):
    """Create a frame container to avoid grid/pack conflicts on root."""
    import customtkinter as ctk

    frame = ctk.CTkFrame(tk_root)
    frame.pack(fill="both", expand=True)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    return frame


@pytest.fixture
def textbox(container):
    """Create a CTkTextbox with sample text."""
    import customtkinter as ctk

    tb = ctk.CTkTextbox(container, wrap="word")
    tb.grid(row=1, column=0, sticky="nsew")
    tb.insert("1.0", "The quick brown fox jumps over the lazy fox.\nFox is a common word.")
    tb.configure(state="disabled")
    return tb


@pytest.fixture
def find_bar(container, textbox):
    """Create a TextFindBar attached to the textbox."""
    from src.ui.text_find_bar import TextFindBar

    bar = TextFindBar(container, textbox)
    bar.grid(row=0, column=0, sticky="ew")
    bar.grid_remove()
    return bar


class TestFindBarShowHide:
    """Test show/hide behavior."""

    def test_show_makes_visible(self, find_bar):
        """show() should grid the bar."""
        find_bar.show()
        # After show, the entry should exist and be focusable
        assert find_bar._entry.winfo_exists()

    def test_hide_clears_state(self, find_bar, textbox):
        """hide() should clear matches and highlights."""
        find_bar.show()
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        assert len(find_bar._matches) > 0

        find_bar.hide()
        assert find_bar._matches == []
        assert find_bar._current_idx == -1

    def test_hide_clears_highlights(self, find_bar, textbox):
        """hide() should remove all highlight tags from the textbox."""
        find_bar.show()
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()

        find_bar.hide()
        # Check that no find_highlight ranges remain
        text_widget = textbox._textbox
        ranges = text_widget.tag_ranges("find_highlight")
        assert len(ranges) == 0


class TestSearchFunctionality:
    """Test search and highlighting."""

    def test_basic_search_finds_matches(self, find_bar):
        """Searching for 'fox' should find 3 matches (case-insensitive)."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        assert len(find_bar._matches) == 3

    def test_search_is_case_insensitive(self, find_bar):
        """'FOX' should match 'fox' and 'Fox'."""
        find_bar._entry.insert(0, "FOX")
        find_bar._do_search()
        assert len(find_bar._matches) == 3

    def test_no_matches_sets_label(self, find_bar):
        """Searching for nonexistent text should show 'No matches'."""
        find_bar._entry.insert(0, "elephant")
        find_bar._do_search()
        assert len(find_bar._matches) == 0
        assert "No matches" in find_bar._count_label.cget("text")

    def test_empty_query_clears(self, find_bar):
        """Empty search clears matches and label."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        assert len(find_bar._matches) == 3

        find_bar._entry.delete(0, "end")
        find_bar._do_search()
        assert len(find_bar._matches) == 0
        assert find_bar._count_label.cget("text") == ""

    def test_search_highlights_all_matches(self, find_bar, textbox):
        """All matches should have the find_highlight tag."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()

        text_widget = textbox._textbox
        ranges = text_widget.tag_ranges("find_highlight")
        # 3 matches = 6 range entries (start, end pairs)
        assert len(ranges) == 6

    def test_current_match_highlighted(self, find_bar, textbox):
        """First match should have the find_current tag after search."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()

        text_widget = textbox._textbox
        current_ranges = text_widget.tag_ranges("find_current")
        assert len(current_ranges) == 2  # One start-end pair

    def test_count_label_shows_position(self, find_bar):
        """Count label should show '1 of 3' after first search."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        assert find_bar._count_label.cget("text") == "1 of 3"


class TestNavigation:
    """Test prev/next match navigation."""

    def test_next_advances_index(self, find_bar):
        """next_match should advance current index."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        assert find_bar._current_idx == 0

        find_bar._next_match()
        assert find_bar._current_idx == 1
        assert find_bar._count_label.cget("text") == "2 of 3"

    def test_next_wraps_around(self, find_bar):
        """next_match should wrap from last to first."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()

        find_bar._next_match()  # 1
        find_bar._next_match()  # 2
        find_bar._next_match()  # wraps to 0
        assert find_bar._current_idx == 0
        assert find_bar._count_label.cget("text") == "1 of 3"

    def test_prev_goes_backward(self, find_bar):
        """prev_match should go backward."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        find_bar._next_match()  # 1

        find_bar._prev_match()  # back to 0
        assert find_bar._current_idx == 0

    def test_prev_wraps_to_last(self, find_bar):
        """prev_match from first should wrap to last match."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()

        find_bar._prev_match()  # wraps to 2
        assert find_bar._current_idx == 2
        assert find_bar._count_label.cget("text") == "3 of 3"

    def test_next_no_op_without_matches(self, find_bar):
        """next/prev should do nothing if no matches."""
        find_bar._next_match()
        assert find_bar._current_idx == -1

        find_bar._prev_match()
        assert find_bar._current_idx == -1


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_char_search(self, find_bar):
        """Single character search should work."""
        find_bar._entry.insert(0, "q")
        find_bar._do_search()
        assert len(find_bar._matches) >= 1

    def test_multiline_content(self, find_bar):
        """Search should work across the second line."""
        find_bar._entry.insert(0, "common")
        find_bar._do_search()
        assert len(find_bar._matches) == 1

    def test_repeated_search_clears_old(self, find_bar, textbox):
        """A new search should clear previous highlights."""
        find_bar._entry.insert(0, "fox")
        find_bar._do_search()
        assert len(find_bar._matches) == 3

        find_bar._entry.delete(0, "end")
        find_bar._entry.insert(0, "quick")
        find_bar._do_search()
        assert len(find_bar._matches) == 1

        text_widget = textbox._textbox
        ranges = text_widget.tag_ranges("find_highlight")
        assert len(ranges) == 2  # Only 1 match = 2 range entries

    def test_whitespace_only_query(self, find_bar):
        """Whitespace-only query should be treated as empty."""
        find_bar._entry.insert(0, "   ")
        find_bar._do_search()
        assert len(find_bar._matches) == 0
        assert find_bar._count_label.cget("text") == ""
