"""
CharacterSanitizer: Clean extracted text for AI model processing.

This module (Step 2.5 of the document pipeline) sanitizes text extracted from
PDFs and OCR, removing/fixing problematic characters that cause Ollama and other
AI models to fail or produce garbled output.

Problems addressed:
1. Mojibake (encoding corruption): ñêcessary → necessary
2. Control characters: non-breaking spaces, zero-width chars, etc.
3. Redacted characters: ██ → [REDACTED]
4. Private-use Unicode: invalid codepoints
5. Malformed UTF-8 sequences from OCR
6. Stray accents: é/ê/à in English text → remove accents for readability

Uses ftfy for encoding recovery + unicodedata for character classification.
Optionally uses unidecode for transliteration (ASCII-safe output).
"""

import re
import time
import unicodedata

import ftfy

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False


class CharacterSanitizer:
    """
    Sanitize text for reliable AI model processing.

    Performs a multi-stage cleanup:
    1. Fix mojibake using ftfy
    2. Normalize Unicode (NFKC form)
    3. Transliterate stray accents (optional, requires unidecode)
    4. Remove/replace control characters
    5. Handle redacted characters
    6. Clean up private-use Unicode
    7. Normalize whitespace
    """

    def __init__(self, preserve_newlines: bool = True, transliterate: bool = True):
        """
        Initialize the sanitizer.

        Args:
            preserve_newlines: If True, keep actual newlines. If False, replace with spaces.
            transliterate: If True (default), convert accented chars (é/ê) to ASCII equivalents.
                         Requires 'unidecode' library. Falls back to ftfy if unavailable.
        """
        self.preserve_newlines = preserve_newlines
        self.transliterate = transliterate and HAS_UNIDECODE
        self.sanitization_log = []

    def sanitize(self, text: str) -> tuple[str, dict]:
        """
        Sanitize text and return cleaned text + statistics.

        6-stage pipeline with comprehensive logging:
        1. Fix mojibake using ftfy
        2. Normalize Unicode (NFKC form)
        3. Transliterate stray accents (optional, requires unidecode)
        4. Handle redacted characters (██ → [REDACTED])
        5. Remove/replace problematic characters
        6. Clean up excessive whitespace

        Args:
            text: Raw extracted text from PDF/OCR

        Returns:
            (cleaned_text, stats_dict) where stats_dict contains:
            - chars_removed: Count of removed characters
            - mojibake_fixed: Count of mojibake fixes
            - control_chars_removed: Count of control characters removed
            - redactions_replaced: Count of redacted characters replaced
            - private_use_removed: Count of private-use chars removed
            - transliterations: Count of accented chars converted to ASCII
        """
        self.sanitization_log = []
        stats = {
            "chars_removed": 0,
            "mojibake_fixed": 0,
            "control_chars_removed": 0,
            "redactions_replaced": 0,
            "private_use_removed": 0,
            "transliterations": 0,
        }

        # Stage 1: Fix mojibake (encoding corruption)
        self._log("Stage 1: Mojibake recovery (ftfy)")
        start = time.time()
        original_len = len(text)
        try:
            text, mojibake_count = self._fix_mojibake(text)
            stats["mojibake_fixed"] = mojibake_count
            duration = time.time() - start
            self._log(f"  ✅ SUCCESS ({duration:.3f}s) - Fixed {mojibake_count} chars")
            self._log(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            self._log(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 2: Normalize Unicode (NFKC form)
        self._log("Stage 2: Unicode normalization (NFKC)")
        start = time.time()
        original_len = len(text)
        try:
            text = self._normalize_unicode(text)
            duration = time.time() - start
            self._log(f"  ✅ SUCCESS ({duration:.3f}s)")
            self._log(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            self._log(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 3: Transliterate stray accents (ê → e, é → e, etc.)
        if self.transliterate:
            self._log("Stage 3: Transliteration (accent conversion)")
            start = time.time()
            original_len = len(text)
            try:
                text, trans_count = self._transliterate_text(text)
                stats["transliterations"] = trans_count
                duration = time.time() - start
                self._log(f"  ✅ SUCCESS ({duration:.3f}s) - Transliterated {trans_count} chars")
                self._log(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
            except Exception as e:
                duration = time.time() - start
                self._log(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
                raise
        else:
            self._log("Stage 3: Transliteration (SKIPPED - disabled)")

        # Stage 4: Handle redacted characters (██ → [REDACTED])
        self._log("Stage 4: Redaction handling")
        start = time.time()
        original_len = len(text)
        try:
            text, redactions = self._handle_redactions(text)
            stats["redactions_replaced"] = redactions
            duration = time.time() - start
            self._log(f"  ✅ SUCCESS ({duration:.3f}s) - Replaced {redactions} redaction chars")
            self._log(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            self._log(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 5: Remove/replace problematic characters
        self._log("Stage 5: Problematic character removal")
        start = time.time()
        original_len = len(text)
        try:
            text, removed, control_removed, private_use = self._clean_problematic_chars(text)
            stats["control_chars_removed"] = control_removed
            stats["private_use_removed"] = private_use
            duration = time.time() - start
            self._log(f"  ✅ SUCCESS ({duration:.3f}s) - Removed {control_removed} control + {private_use} private-use chars")
            self._log(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            self._log(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        # Stage 6: Clean up excessive whitespace
        self._log("Stage 6: Whitespace normalization")
        start = time.time()
        original_len = len(text)
        try:
            text = self._clean_whitespace(text)
            duration = time.time() - start
            self._log(f"  ✅ SUCCESS ({duration:.3f}s)")
            self._log(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
        except Exception as e:
            duration = time.time() - start
            self._log(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
            raise

        stats["chars_removed"] = removed + control_removed + private_use

        return text, stats

    def _fix_mojibake(self, text: str) -> tuple[str, int]:
        """
        Fix mojibake (encoding corruption) using ftfy.

        Examples:
            'ñêcessary' → 'necessary'
            'dccedêñt' → 'decedent'
            'Defeñdañt' → 'Defendant'
        """
        original = text
        text = ftfy.fix_text(text)

        # Count changes by comparing character counts
        # Note: ftfy might also add/remove chars, so count actual mojibake fixes
        # strict=False: strings may have different lengths after ftfy processing
        fixes = sum(1 for a, b in zip(original, text, strict=False) if a != b)

        if fixes > 0:
            self._log(f"Fixed {fixes} mojibake/encoding corruption characters")

        return text, fixes

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode to NFKC form.

        NFKC (Compatibility Decomposition + Canonical Composition):
        - Decomposes characters to base forms
        - Recomposes into canonical form
        - Handles ligatures, superscripts, etc.
        """
        original_len = len(text)
        text = unicodedata.normalize('NFKC', text)

        if len(text) != original_len:
            self._log(f"Unicode normalization: {original_len} → {len(text)} chars")

        return text

    def _transliterate_text(self, text: str) -> tuple[str, int]:
        """
        Transliterate accented characters to ASCII equivalents.

        Examples:
            'ñêcessary' → 'necessary' (both ñ and ê become their base forms)
            'locãted' → 'located'
            'dccedêñt' → 'dccedent'

        This fixes corruption where OCR/extraction introduces stray accents.
        Uses unidecode to convert all accented chars to ASCII.
        """
        if not HAS_UNIDECODE:
            return text, 0

        original = text
        text = unidecode(text)

        # Count changes (strict=False: unidecode may change string length)
        transliterations = sum(1 for a, b in zip(original, text, strict=False) if a != b)

        if transliterations > 0:
            self._log(f"Transliterated {transliterations} accented characters to ASCII")

        return text, transliterations

    def _handle_redactions(self, text: str) -> tuple[str, int]:
        """
        Replace redacted characters (██) with [REDACTED] marker.

        Common redaction patterns:
        - ██ (U+2588 FULL BLOCK repeated)
        - ▓▓ (U+2593 DARK SHADE repeated)
        - ████ (longer sequences)
        """
        original = text

        # Replace sequences of redaction blocks with marker
        text = re.sub(r'(█{2,})', ' [REDACTED] ', text)
        text = re.sub(r'(▓{2,})', ' [REDACTED] ', text)
        text = re.sub(r'(▒{2,})', ' [REDACTED] ', text)

        redactions = original.count('█') + original.count('▓') + original.count('▒')

        if redactions > 0:
            self._log(f"Replaced {redactions} redaction characters with [REDACTED] markers")

        return text, redactions

    def _clean_problematic_chars(self, text: str) -> tuple[str, int, int, int]:
        """
        Remove or replace problematic characters.

        Returns:
            (cleaned_text, other_replaced_count, control_removed_count, private_use_count)
        """
        cleaned = []
        control_removed = 0
        private_use_removed = 0
        other_replaced = 0

        for char in text:
            category = unicodedata.category(char)

            # Control characters (C* category)
            if category[0] == 'C':
                # Special handling for newlines and tabs (preserve if desired)
                if char in '\n\t':
                    # Keep newlines/tabs, they're useful for structure
                    cleaned.append(char)
                else:
                    # Remove other control chars (spaces, format chars, etc.)
                    control_removed += 1

                    # Replace with space for readability (except for invisible chars)
                    if category == 'Cc':  # Control characters
                        cleaned.append(' ')
                    # Skip format characters (Cf) entirely
                    # Skip other C-category chars entirely

            # Private-use characters (Co category)
            elif category == 'Co':
                private_use_removed += 1
                cleaned.append(' ')

            # Surrogate characters (Cs category) - malformed UTF-8
            elif category == 'Cs':
                private_use_removed += 1
                cleaned.append('?')

            # Other problematic characters
            # Zero-width characters, combining marks that appear corrupted
            elif char in '\u200b\u200c\u200d\ufeff':  # Zero-width space, ZWJ, BOM, etc.
                other_replaced += 1
                cleaned.append(' ')

            # Keep everything else
            else:
                cleaned.append(char)

        text = ''.join(cleaned)

        if control_removed > 0:
            self._log(f"Removed {control_removed} control characters")
        if private_use_removed > 0:
            self._log(f"Removed {private_use_removed} private-use/surrogate characters")
        if other_replaced > 0:
            self._log(f"Replaced {other_replaced} zero-width characters with spaces")

        return text, other_replaced, control_removed, private_use_removed

    def _clean_whitespace(self, text: str) -> str:
        """
        Clean up excessive whitespace while preserving document structure.

        - Replace multiple spaces with single space
        - Replace multiple blank lines with double newline (paragraph break)
        - Preserve leading/trailing newlines for document integrity
        """
        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Replace non-breaking spaces and similar with regular spaces
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u2000', ' ')  # En quad
        text = text.replace('\u2001', ' ')  # Em quad

        # Clean up multiple spaces (but not newlines)
        text = re.sub(r' {2,}', ' ', text)

        # Clean up multiple blank lines (preserve max 2 newlines = 1 blank line)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _log(self, message: str) -> None:
        """Log sanitization actions for debugging."""
        self.sanitization_log.append(message)

    def get_log(self) -> list[str]:
        """Return the sanitization log."""
        return self.sanitization_log.copy()

    @staticmethod
    def example_usage() -> None:
        """Show example usage of CharacterSanitizer."""
        print("""
        # Usage Example:

        from src.core.sanitization import CharacterSanitizer

        # Create sanitizer
        sanitizer = CharacterSanitizer()

        # Process extracted text
        cleaned_text, stats = sanitizer.sanitize(raw_extracted_text)

        # Check what was fixed
        print(f"Mojibake fixed: {stats['mojibake_fixed']}")
        print(f"Control chars removed: {stats['control_chars_removed']}")
        print(f"Redactions replaced: {stats['redactions_replaced']}")

        # See detailed log
        for log_entry in sanitizer.get_log():
            print(f"  - {log_entry}")

        # Use cleaned text for AI processing
        summary = ollama_model.generate(cleaned_text)
        """)
