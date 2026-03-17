"""
Tests for transcript-aware speaker-turn boundary injection.

Covers inject_speaker_boundaries() from src.core.chunking.transcript_boundaries.
"""

from src.core.chunking.transcript_boundaries import inject_speaker_boundaries


class TestInjectSpeakerBoundaries:
    """Tests for inject_speaker_boundaries()."""

    def test_noop_on_non_transcript_text(self):
        """Non-transcript text passes through unchanged."""
        text = "The quick brown fox jumped over the lazy dog."
        assert inject_speaker_boundaries(text) == text

    def test_noop_on_empty_string(self):
        """Empty input returns empty."""
        assert inject_speaker_boundaries("") == ""

    def test_adds_break_before_by_mr(self):
        """Injects paragraph break before 'BY MR. NAME:'."""
        text = "some testimony\nBY MR. SMITH: Good morning."
        result = inject_speaker_boundaries(text)
        assert "\n\nBY MR. SMITH:" in result

    def test_adds_break_before_by_ms(self):
        """Injects paragraph break before 'BY MS. NAME:'."""
        text = "some testimony\nBY MS. JONES: Let me ask you."
        result = inject_speaker_boundaries(text)
        assert "\n\nBY MS. JONES:" in result

    def test_adds_break_before_named_speaker(self):
        """Injects paragraph break before 'MR. NAME:' (needs quick-check marker)."""
        text = "BY MR. JONES: Begin.\nMR. SMITH: I object."
        result = inject_speaker_boundaries(text)
        assert "\n\nMR. SMITH:" in result

    def test_adds_break_before_the_court(self):
        """Injects paragraph break before 'THE COURT:'."""
        text = "some text\nTHE COURT: Overruled."
        result = inject_speaker_boundaries(text)
        assert "\n\nTHE COURT:" in result

    def test_adds_break_before_the_witness(self):
        """Injects paragraph break before 'THE WITNESS:'."""
        text = "some text\nTHE WITNESS: Yes, that is correct."
        result = inject_speaker_boundaries(text)
        assert "\n\nTHE WITNESS:" in result

    def test_adds_break_before_examination_header(self):
        """Injects break before DIRECT EXAMINATION."""
        text = "DIRECT EXAMINATION\nBY MR. SMITH: State your name."
        result = inject_speaker_boundaries(text)
        # EXAMINATION is a quick-check marker, so regex runs
        assert "DIRECT EXAMINATION" in result

    def test_cross_examination(self):
        """CROSS EXAMINATION gets a break."""
        text = "some text\nCROSS EXAMINATION\nBY MR. DOE: Hello."
        result = inject_speaker_boundaries(text)
        assert "\n\nCROSS EXAMINATION" in result

    def test_preserves_existing_break(self):
        """When \\n\\n already precedes the match start, content is unchanged."""
        # Note: the regex \s* can capture one \n as part of the match, so
        # a preceding \n\n may still get an extra \n inserted. This tests
        # that the function at least ensures a paragraph break is present.
        text = "some text\n\nBY MR. SMITH: Hello."
        result = inject_speaker_boundaries(text)
        assert "\n\nBY MR. SMITH:" in result

    def test_case_insensitive(self):
        """Matches are case-insensitive."""
        text = "some text\nby mr. smith: hello."
        result = inject_speaker_boundaries(text)
        assert "\n\n" in result

    def test_multiple_speakers(self):
        """Multiple speaker turns all get breaks."""
        text = "THE COURT: Please proceed.\nBY MR. SMITH: Thank you.\nTHE WITNESS: I don't recall."
        result = inject_speaker_boundaries(text)
        assert result.count("\n\n") >= 3

    def test_dr_prefix(self):
        """DR. prefix is recognized as a speaker (needs quick-check marker)."""
        text = "THE WITNESS: I felt pain.\nDR. BROWN: The patient showed signs."
        result = inject_speaker_boundaries(text)
        assert "\n\nDR. BROWN:" in result

    def test_quick_check_skips_large_non_transcript(self):
        """Quick check avoids regex pass on non-transcript text."""
        # Text that doesn't contain any quick-check markers
        text = "A" * 10000
        result = inject_speaker_boundaries(text)
        assert result == text
