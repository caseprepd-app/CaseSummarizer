"""
Document text pagination utility.

Splits large document text into manageable sections for display,
preventing GUI freezes from inserting thousands of words at once.
"""


def split_into_sections(text: str, words_per_section: int = 300) -> list[str]:
    """
    Split text into ~N-word sections, breaking at paragraph boundaries.

    Accumulates paragraphs until word count exceeds the threshold, then
    starts a new section. Single paragraphs longer than the threshold
    are force-split at word boundaries.

    Args:
        text: Full document text
        words_per_section: Target words per section (default 300)

    Returns:
        List of text sections (at least one, even if empty)
    """
    if not text or not text.strip():
        return [text or ""]

    total_words = len(text.split())
    if total_words <= words_per_section:
        return [text]

    paragraphs = text.split("\n")
    sections = []
    current_lines = []
    current_count = 0

    for para in paragraphs:
        para_words = len(para.split()) if para.strip() else 0

        # Force-split a single paragraph that exceeds the threshold
        if para_words > words_per_section and not current_lines:
            words = para.split()
            for i in range(0, len(words), words_per_section):
                sections.append(" ".join(words[i : i + words_per_section]))
            continue

        # Adding this paragraph would exceed threshold — flush
        if current_count + para_words > words_per_section and current_lines:
            sections.append("\n".join(current_lines))
            current_lines = []
            current_count = 0

        current_lines.append(para)
        current_count += para_words

    # Flush remaining
    if current_lines:
        sections.append("\n".join(current_lines))

    return sections if sections else [text]
