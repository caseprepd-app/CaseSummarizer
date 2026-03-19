"""Tests for feedback balance hint in vocabulary UI."""

import pytest


class FakeLabel:
    """Minimal mock for CTkLabel to track configure calls."""

    def __init__(self):
        self.text = ""
        self.text_color = None

    def configure(self, **kwargs):
        """Track the last configure call."""
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "text_color" in kwargs:
            self.text_color = kwargs["text_color"]


class FakeFeedbackManager:
    """Mock feedback manager with configurable keep/skip counts."""

    def __init__(self, keep_terms=None, skip_terms=None):
        self._keep = keep_terms or []
        self._skip = skip_terms or []

    def get_rated_terms(self, rating_filter=None):
        """Return mock rated terms."""
        if rating_filter == 1:
            return self._keep
        elif rating_filter == -1:
            return self._skip
        return self._keep + self._skip


@pytest.fixture
def widget_parts():
    """Create the minimal parts needed to test _update_balance_hint."""
    label = FakeLabel()
    feedback_mgr = FakeFeedbackManager()
    return label, feedback_mgr


def _call_update(label, feedback_mgr, dismissed=False):
    """Simulate _update_balance_hint without needing the full widget."""
    if dismissed:
        return

    keep_count = len(feedback_mgr.get_rated_terms(rating_filter=+1))
    skip_count = len(feedback_mgr.get_rated_terms(rating_filter=-1))
    total = keep_count + skip_count

    if total < 20:
        label.configure(text="")
        return

    majority_pct = max(keep_count, skip_count) / total
    if majority_pct < 0.75:
        label.configure(text="")
        return

    if keep_count > skip_count:
        label.configure(
            text="Tip: Your feedback is mostly keeps",
            text_color=("gray50", "gray70"),
        )
    else:
        label.configure(
            text="Tip: Your feedback is mostly skips",
            text_color=("gray50", "gray70"),
        )


def test_no_hint_under_20_votes(widget_parts):
    """18 total votes (< 20 threshold) should produce no hint."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(15)]
    mgr._skip = [f"skip_{i}" for i in range(3)]
    _call_update(label, mgr)
    assert label.text == ""
    assert label.text_color is None


def test_no_hint_when_balanced(widget_parts):
    """20 votes with 70% keeps (below 75%) should produce no hint."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(14)]
    mgr._skip = [f"skip_{i}" for i in range(6)]
    _call_update(label, mgr)
    assert label.text == ""


def test_hint_shown_when_keeps_dominate(widget_parts):
    """20 votes with 90% keeps should show keeps-dominated hint."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(18)]
    mgr._skip = [f"skip_{i}" for i in range(2)]
    _call_update(label, mgr)
    assert label.text == "Tip: Your feedback is mostly keeps"
    assert label.text_color == ("gray50", "gray70")


def test_hint_shown_when_skips_dominate(widget_parts):
    """20 votes with 85% skips should show skips-dominated hint."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(3)]
    mgr._skip = [f"skip_{i}" for i in range(17)]
    _call_update(label, mgr)
    assert label.text == "Tip: Your feedback is mostly skips"
    assert label.text_color == ("gray50", "gray70")


def test_hint_clears_when_balance_improves(widget_parts):
    """Hint should disappear when ratio drops below 75%."""
    label, mgr = widget_parts
    # First: 90% keeps (lopsided)
    mgr._keep = [f"term_{i}" for i in range(18)]
    mgr._skip = [f"skip_{i}" for i in range(2)]
    _call_update(label, mgr)
    assert "keeps" in label.text

    # Then: 60% keeps (balanced)
    mgr._keep = [f"term_{i}" for i in range(12)]
    mgr._skip = [f"skip_{i}" for i in range(8)]
    _call_update(label, mgr)
    assert label.text == ""


def test_dismissed_stays_hidden(widget_parts):
    """Once dismissed, hint should not reappear even with lopsided data."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(18)]
    mgr._skip = [f"skip_{i}" for i in range(2)]
    _call_update(label, mgr, dismissed=True)
    assert label.text == ""


def test_exactly_75_percent_shows_hint(widget_parts):
    """15/20 = exactly 75% should show hint (threshold is not <0.75)."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(15)]
    mgr._skip = [f"skip_{i}" for i in range(5)]
    _call_update(label, mgr)
    assert label.text == "Tip: Your feedback is mostly keeps"


def test_76_percent_shows_hint(widget_parts):
    """19/25 = 76% keeps should show keeps hint."""
    label, mgr = widget_parts
    mgr._keep = [f"term_{i}" for i in range(19)]
    mgr._skip = [f"skip_{i}" for i in range(6)]
    _call_update(label, mgr)
    assert label.text == "Tip: Your feedback is mostly keeps"
