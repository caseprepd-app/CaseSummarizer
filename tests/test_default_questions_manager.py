"""
Tests for DefaultQuestionsManager.

Covers the persistent question store used by semantic search:
- DefaultQuestion dataclass: to_dict, from_dict
- DefaultQuestionsManager construction and JSON loading
- get_all_questions, get_enabled_questions, get_enabled_count, get_total_count
- set_enabled, add_question, remove_question, update_question, move_question
- replace_all: batch replacement in one disk write
- _create_default_questions: built-in fallback questions
- _migrate_from_legacy: migration from plain text file format
- _load_from_json: error handling with corrupt JSON
- reload: re-reads from disk
- get_default_questions_manager: singleton accessor
"""

import json


class TestDefaultQuestion:
    """Tests for the DefaultQuestion dataclass."""

    def test_default_enabled_is_true(self):
        """Questions are enabled by default."""
        from src.core.semantic.default_questions_manager import DefaultQuestion

        q = DefaultQuestion(text="Who are the plaintiffs?")
        assert q.enabled is True

    def test_explicit_disabled(self):
        """enabled=False is stored correctly."""
        from src.core.semantic.default_questions_manager import DefaultQuestion

        q = DefaultQuestion(text="Any damages?", enabled=False)
        assert q.enabled is False

    def test_to_dict_round_trips(self):
        """to_dict produces a dict that from_dict can reconstruct."""
        from src.core.semantic.default_questions_manager import DefaultQuestion

        original = DefaultQuestion(text="What is the case about?", enabled=True)
        d = original.to_dict()
        restored = DefaultQuestion.from_dict(d)

        assert restored.text == original.text
        assert restored.enabled == original.enabled

    def test_from_dict_defaults_enabled_to_true(self):
        """from_dict treats missing 'enabled' key as True."""
        from src.core.semantic.default_questions_manager import DefaultQuestion

        q = DefaultQuestion.from_dict({"text": "When did it happen?"})
        assert q.enabled is True

    def test_from_dict_preserves_disabled_flag(self):
        """from_dict correctly reads enabled=False."""
        from src.core.semantic.default_questions_manager import DefaultQuestion

        q = DefaultQuestion.from_dict({"text": "Damages?", "enabled": False})
        assert q.enabled is False


class TestDefaultQuestionsManagerLoading:
    """Tests for manager construction and JSON persistence."""

    def test_loads_questions_from_json_file(self, tmp_path):
        """Manager loads questions from a JSON file at config_path."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        json_path = tmp_path / "questions.json"
        json_path.write_text(
            json.dumps(
                {
                    "questions": [
                        {"text": "Who are the parties?", "enabled": True},
                        {"text": "What happened?", "enabled": False},
                    ]
                }
            )
        )

        manager = DefaultQuestionsManager(config_path=json_path)

        assert manager.get_total_count() == 2
        assert manager.get_enabled_count() == 1

    def test_creates_default_questions_when_no_file(self, tmp_path):
        """Manager creates the default question set when no file or legacy file exists."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        json_path = tmp_path / "questions.json"
        manager = DefaultQuestionsManager(config_path=json_path)

        # The default set has 5 questions, all enabled
        assert manager.get_total_count() >= 3
        assert manager.get_enabled_count() == manager.get_total_count()
        # The file was persisted
        assert json_path.exists()

    def test_migrates_from_legacy_text_file(self, tmp_path, monkeypatch):
        """Manager migrates question list from the legacy .txt format."""
        import src.core.semantic.default_questions_manager as dqm_module
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        legacy_path = tmp_path / "legacy_questions.txt"
        legacy_path.write_text("# comment line\nWho filed the case?\nWhat is the claim?\n")

        json_path = tmp_path / "questions.json"

        monkeypatch.setattr(dqm_module, "LEGACY_QUESTIONS_PATH", legacy_path)

        manager = DefaultQuestionsManager(config_path=json_path)

        texts = manager.get_enabled_questions()
        assert "Who filed the case?" in texts
        assert "What is the claim?" in texts
        # Comment lines and blank lines are excluded
        assert any(t.startswith("#") for t in texts) is False
        # JSON file was created during migration
        assert json_path.exists()

    def test_recovers_from_corrupt_json(self, tmp_path):
        """Manager falls back to default questions when JSON is malformed."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        json_path = tmp_path / "questions.json"
        json_path.write_text("{ this is not valid json }")

        manager = DefaultQuestionsManager(config_path=json_path)

        # Falls back to built-in defaults (currently 1 default question)
        assert manager.get_total_count() >= 1

    def test_reload_re_reads_from_disk(self, tmp_path):
        """reload() picks up changes made to the file after construction."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        json_path = tmp_path / "questions.json"
        json_path.write_text(
            json.dumps({"questions": [{"text": "First question?", "enabled": True}]})
        )

        manager = DefaultQuestionsManager(config_path=json_path)
        assert manager.get_total_count() == 1

        # Simulate an external edit
        json_path.write_text(
            json.dumps(
                {
                    "questions": [
                        {"text": "First question?", "enabled": True},
                        {"text": "New question?", "enabled": True},
                    ]
                }
            )
        )

        manager.reload()

        assert manager.get_total_count() == 2


class TestDefaultQuestionsManagerReadMethods:
    """Tests for get_all_questions, get_enabled_questions, counts."""

    def _make_manager(self, tmp_path, questions_data):
        """Create a manager loaded from a given questions list."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        json_path = tmp_path / "q.json"
        json_path.write_text(json.dumps({"questions": questions_data}))
        return DefaultQuestionsManager(config_path=json_path)

    def test_get_all_questions_returns_all_regardless_of_enabled(self, tmp_path):
        """get_all_questions returns both enabled and disabled questions."""
        manager = self._make_manager(
            tmp_path,
            [{"text": "A?", "enabled": True}, {"text": "B?", "enabled": False}],
        )

        all_qs = manager.get_all_questions()
        assert len(all_qs) == 2

    def test_get_enabled_questions_returns_only_enabled_texts(self, tmp_path):
        """get_enabled_questions returns text strings for enabled questions only."""
        manager = self._make_manager(
            tmp_path,
            [{"text": "Enabled?", "enabled": True}, {"text": "Disabled?", "enabled": False}],
        )

        enabled = manager.get_enabled_questions()
        assert "Enabled?" in enabled
        assert "Disabled?" not in enabled

    def test_get_enabled_count_matches_enabled_questions(self, tmp_path):
        """get_enabled_count matches len(get_enabled_questions())."""
        manager = self._make_manager(
            tmp_path,
            [
                {"text": "Q1", "enabled": True},
                {"text": "Q2", "enabled": True},
                {"text": "Q3", "enabled": False},
            ],
        )

        assert manager.get_enabled_count() == 2
        assert manager.get_enabled_count() == len(manager.get_enabled_questions())

    def test_get_total_count_includes_disabled(self, tmp_path):
        """get_total_count includes all questions, enabled and disabled."""
        manager = self._make_manager(
            tmp_path,
            [{"text": "A", "enabled": True}, {"text": "B", "enabled": False}],
        )

        assert manager.get_total_count() == 2

    def test_get_all_questions_returns_copy(self, tmp_path):
        """Modifying the returned list does not affect the manager's internal state."""
        manager = self._make_manager(tmp_path, [{"text": "Q?", "enabled": True}])

        result = manager.get_all_questions()
        result.clear()

        assert manager.get_total_count() == 1


class TestDefaultQuestionsManagerMutations:
    """Tests for set_enabled, add_question, remove_question, update_question, move_question."""

    def _make_manager(self, tmp_path, texts=None):
        """Create a manager with simple enabled questions."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        data = [
            {"text": t, "enabled": True}
            for t in (texts if texts is not None else ["Q1?", "Q2?", "Q3?"])
        ]
        json_path = tmp_path / "q.json"
        json_path.write_text(json.dumps({"questions": data}))
        return DefaultQuestionsManager(config_path=json_path)

    def test_set_enabled_disables_question(self, tmp_path):
        """set_enabled(0, False) disables the first question."""
        manager = self._make_manager(tmp_path)

        manager.set_enabled(0, False)

        assert not manager.get_all_questions()[0].enabled

    def test_set_enabled_persists_to_disk(self, tmp_path):
        """set_enabled writes the change to the JSON file."""
        manager = self._make_manager(tmp_path)
        manager.set_enabled(0, False)

        # Load fresh instance from same file
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        manager2 = DefaultQuestionsManager(config_path=tmp_path / "q.json")
        assert not manager2.get_all_questions()[0].enabled

    def test_set_enabled_ignores_out_of_range_index(self, tmp_path):
        """set_enabled with an invalid index is silently ignored."""
        manager = self._make_manager(tmp_path)
        manager.set_enabled(99, False)  # Should not raise

    def test_add_question_appends_to_list(self, tmp_path):
        """add_question adds the new question at the end."""
        manager = self._make_manager(tmp_path, ["Existing?"])

        idx = manager.add_question("New question?")

        assert idx == 1
        assert manager.get_total_count() == 2
        assert "New question?" in manager.get_enabled_questions()

    def test_add_question_trims_whitespace(self, tmp_path):
        """add_question strips leading/trailing whitespace from the text."""
        manager = self._make_manager(tmp_path, [])

        manager.add_question("  Trimmed question?  ")

        texts = [q.text for q in manager.get_all_questions()]
        assert "Trimmed question?" in texts

    def test_add_question_rejects_empty_text(self, tmp_path):
        """add_question returns -1 and does not add when text is empty/whitespace."""
        manager = self._make_manager(tmp_path, [])

        result = manager.add_question("   ")

        assert result == -1
        assert manager.get_total_count() == 0

    def test_remove_question_removes_by_index(self, tmp_path):
        """remove_question removes the question at the given index."""
        manager = self._make_manager(tmp_path, ["Keep?", "Remove?", "Keep too?"])

        success = manager.remove_question(1)

        assert success is True
        assert manager.get_total_count() == 2
        texts = manager.get_enabled_questions()
        assert "Remove?" not in texts
        assert "Keep?" in texts

    def test_remove_question_returns_false_for_bad_index(self, tmp_path):
        """remove_question returns False and changes nothing for an out-of-range index."""
        manager = self._make_manager(tmp_path, ["Only?"])

        result = manager.remove_question(5)

        assert result is False
        assert manager.get_total_count() == 1

    def test_update_question_changes_text(self, tmp_path):
        """update_question replaces the text of the question at the given index."""
        manager = self._make_manager(tmp_path, ["Old text?"])

        success = manager.update_question(0, "New text?")

        assert success is True
        assert manager.get_all_questions()[0].text == "New text?"

    def test_update_question_returns_false_for_empty_text(self, tmp_path):
        """update_question returns False when new text is empty."""
        manager = self._make_manager(tmp_path, ["Existing?"])

        result = manager.update_question(0, "   ")

        assert result is False
        assert manager.get_all_questions()[0].text == "Existing?"

    def test_update_question_returns_false_for_bad_index(self, tmp_path):
        """update_question returns False for an out-of-range index."""
        manager = self._make_manager(tmp_path)

        assert manager.update_question(99, "Text?") is False

    def test_move_question_reorders_list(self, tmp_path):
        """move_question moves a question from one position to another."""
        manager = self._make_manager(tmp_path, ["A?", "B?", "C?"])

        success = manager.move_question(0, 2)  # Move A from position 0 to position 2

        assert success is True
        texts = [q.text for q in manager.get_all_questions()]
        assert texts.index("A?") == 2

    def test_move_question_same_position_is_noop(self, tmp_path):
        """move_question with from_index == to_index returns True without changing order."""
        manager = self._make_manager(tmp_path, ["X?", "Y?"])

        result = manager.move_question(0, 0)

        assert result is True
        assert manager.get_all_questions()[0].text == "X?"

    def test_move_question_returns_false_for_bad_index(self, tmp_path):
        """move_question returns False for out-of-range from_index or to_index."""
        manager = self._make_manager(tmp_path, ["Only?"])

        assert manager.move_question(5, 0) is False
        assert manager.move_question(0, 5) is False


class TestReplaceAll:
    """Tests for DefaultQuestionsManager.replace_all."""

    def _make_manager(self, tmp_path):
        """Create an empty manager."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        json_path = tmp_path / "q.json"
        json_path.write_text(json.dumps({"questions": [{"text": "Old?", "enabled": True}]}))
        return DefaultQuestionsManager(config_path=json_path)

    def test_replaces_all_questions(self, tmp_path):
        """replace_all swaps entire question list with the provided data."""
        manager = self._make_manager(tmp_path)

        manager.replace_all(
            [
                {"text": "New Q1?", "enabled": True},
                {"text": "New Q2?", "enabled": False},
            ]
        )

        assert manager.get_total_count() == 2
        texts = [q.text for q in manager.get_all_questions()]
        assert "New Q1?" in texts
        assert "New Q2?" in texts
        assert "Old?" not in texts

    def test_replace_all_skips_empty_texts(self, tmp_path):
        """replace_all skips entries where text is empty or whitespace."""
        manager = self._make_manager(tmp_path)

        manager.replace_all(
            [
                {"text": "Valid?", "enabled": True},
                {"text": "  ", "enabled": True},  # should be skipped
                {"text": "", "enabled": True},  # should be skipped
            ]
        )

        assert manager.get_total_count() == 1
        assert manager.get_enabled_questions()[0] == "Valid?"

    def test_replace_all_persists_to_disk(self, tmp_path):
        """replace_all writes the new question set to the JSON file."""
        from src.core.semantic.default_questions_manager import DefaultQuestionsManager

        manager = self._make_manager(tmp_path)
        manager.replace_all([{"text": "Persisted?", "enabled": True}])

        # Fresh load from disk
        manager2 = DefaultQuestionsManager(config_path=tmp_path / "q.json")
        assert manager2.get_enabled_questions() == ["Persisted?"]


class TestDefaultQuestionsManagerSingleton:
    """Tests for get_default_questions_manager singleton accessor."""

    def test_returns_manager_instance(self):
        """get_default_questions_manager returns a DefaultQuestionsManager."""
        from src.core.semantic.default_questions_manager import (
            DefaultQuestionsManager,
            get_default_questions_manager,
        )

        manager = get_default_questions_manager()
        assert isinstance(manager, DefaultQuestionsManager)

    def test_returns_same_instance_on_repeated_calls(self):
        """Repeated calls return the same singleton instance."""
        from src.core.semantic.default_questions_manager import get_default_questions_manager

        m1 = get_default_questions_manager()
        m2 = get_default_questions_manager()
        assert m1 is m2
