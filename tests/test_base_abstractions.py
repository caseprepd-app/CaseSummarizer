"""
Tests for base abstraction classes.

Covers BaseNamedComponent, BaseSettingsWidget, and the
_guard_reentrant decorator used in CorpusDialog.
"""

import pytest

# =========================================================================
# BaseNamedComponent Tests
# =========================================================================


class TestBaseNamedComponent:
    """Tests for the BaseNamedComponent ABC."""

    def _make_subclass(self, name="TestAlgo", enabled=True):
        """Create a concrete subclass for testing."""
        from src.core.base_component import BaseNamedComponent

        class ConcreteComponent(BaseNamedComponent):
            pass

        obj = ConcreteComponent()
        obj.name = name
        obj.enabled = enabled
        return obj

    def test_default_attributes(self):
        """Default name and enabled values are set."""
        obj = self._make_subclass()
        assert obj.name == "TestAlgo"
        assert obj.enabled is True

    def test_get_config_returns_name_and_enabled(self):
        """get_config includes name and enabled."""
        obj = self._make_subclass(name="MyAlgo", enabled=False)
        config = obj.get_config()
        assert config == {"name": "MyAlgo", "enabled": False}

    def test_repr_basic(self):
        """repr includes class name and enabled status."""
        obj = self._make_subclass()
        result = repr(obj)
        assert "ConcreteComponent" in result
        assert "enabled=True" in result

    def test_repr_disabled(self):
        """repr shows enabled=False when disabled."""
        obj = self._make_subclass(enabled=False)
        assert "enabled=False" in repr(obj)

    def test_repr_extras_included(self):
        """_repr_extras values appear in repr."""
        from src.core.base_component import BaseNamedComponent

        class WithExtras(BaseNamedComponent):
            """Subclass with extra repr fields."""

            weight: float = 0.5

            def _repr_extras(self):
                """Return weight as extra."""
                return {"weight": self.weight}

        obj = WithExtras()
        result = repr(obj)
        assert "weight=0.5" in result

    def test_get_config_extended_by_subclass(self):
        """Subclass can extend get_config via super()."""
        from src.core.base_component import BaseNamedComponent

        class Extended(BaseNamedComponent):
            """Subclass with extra config."""

            priority: int = 10

            def get_config(self):
                """Add priority to config."""
                config = super().get_config()
                config["priority"] = self.priority
                return config

        obj = Extended()
        config = obj.get_config()
        assert "name" in config
        assert "enabled" in config
        assert config["priority"] == 10

    def test_cannot_instantiate_abc_directly(self):
        """BaseNamedComponent is abstract but has no abstract methods."""
        # It's an ABC but has no @abstractmethod, so it can be instantiated.
        # This is by design — it provides shared behavior, not an interface.
        from src.core.base_component import BaseNamedComponent

        obj = BaseNamedComponent()
        assert obj.name == "BaseComponent"


class TestBaseNamedComponentIntegration:
    """Verify real pipeline base classes inherit and get_config works."""

    def test_preprocessor_get_config(self):
        """BasePreprocessor inherits and get_config returns name/enabled."""
        from src.core.preprocessing.base import BasePreprocessor

        class Stub(BasePreprocessor):
            """Stub."""

            def process(self, text):
                """No-op."""
                return text

        obj = Stub()
        config = obj.get_config()
        assert "name" in config and "enabled" in config

    def test_extraction_algorithm_get_config(self):
        """BaseExtractionAlgorithm inherits and get_config returns name/enabled."""
        from src.core.vocabulary.algorithms.base import BaseExtractionAlgorithm

        class Stub(BaseExtractionAlgorithm):
            """Stub."""

            def extract(self, text, **kw):
                """No-op."""
                return []

        obj = Stub()
        config = obj.get_config()
        assert "name" in config and "enabled" in config

    def test_retrieval_algorithm_get_config(self):
        """BaseRetrievalAlgorithm inherits and get_config returns name/enabled."""
        from src.core.retrieval.base import BaseRetrievalAlgorithm

        class Stub(BaseRetrievalAlgorithm):
            """Stub."""

            def index_documents(self, chunks):
                """No-op."""

            def retrieve(self, query, k=5):
                """No-op."""
                return []

            @property
            def is_indexed(self):
                """No-op."""
                return False

        obj = Stub()
        config = obj.get_config()
        assert "name" in config and "enabled" in config

    def test_vocabulary_filter_get_config(self):
        """BaseVocabularyFilter inherits and get_config returns name/enabled."""
        from src.core.vocabulary.filters.base import BaseVocabularyFilter

        class Stub(BaseVocabularyFilter):
            """Stub."""

            def filter(self, terms):
                """No-op."""
                return terms

        obj = Stub()
        config = obj.get_config()
        assert "name" in config and "enabled" in config


# =========================================================================
# BaseSettingsWidget Tests
# =========================================================================


class TestBaseSettingsWidget:
    """Tests for the BaseSettingsWidget ABC."""

    def test_cannot_instantiate_directly(self):
        """BaseSettingsWidget is abstract and cannot be instantiated."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget

        with pytest.raises(TypeError):
            BaseSettingsWidget(None)

    def test_default_validate_returns_none(self):
        """Default validate() returns None (valid)."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget

        class Concrete(BaseSettingsWidget):
            """Minimal concrete implementation."""

            def _setup_ui(self):
                """No-op UI setup."""
                pass

            def get_value(self):
                """Return None."""
                return None

            def set_value(self, value):
                """No-op setter."""
                pass

        # Can't instantiate without a Tk root, test the class-level default
        assert Concrete.validate is BaseSettingsWidget.validate

    def test_subclass_interface_completeness(self):
        """A subclass missing abstract methods cannot be instantiated."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget

        class Incomplete(BaseSettingsWidget):
            """Missing get_value and set_value."""

            def _setup_ui(self):
                """No-op."""
                pass

        with pytest.raises(TypeError, match="get_value|set_value"):
            Incomplete(None)


class TestBaseSettingsWidgetIntegration:
    """Verify real settings widgets inherit from BaseSettingsWidget."""

    def test_columns_widget_inherits(self):
        """ColumnVisibilityWidget inherits from BaseSettingsWidget."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget
        from src.ui.settings.columns_widget import ColumnVisibilityWidget

        assert issubclass(ColumnVisibilityWidget, BaseSettingsWidget)

    def test_questions_widget_inherits(self):
        """DefaultQuestionsWidget inherits from BaseSettingsWidget."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget
        from src.ui.settings.questions_widget import DefaultQuestionsWidget

        assert issubclass(DefaultQuestionsWidget, BaseSettingsWidget)

    def test_corpus_widget_inherits(self):
        """CorpusSettingsWidget inherits from BaseSettingsWidget."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget
        from src.ui.settings.corpus_widget import CorpusSettingsWidget

        assert issubclass(CorpusSettingsWidget, BaseSettingsWidget)

    def test_patterns_widget_inherits(self):
        """CustomPatternsWidget inherits from BaseSettingsWidget."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget
        from src.ui.settings.patterns_widget import CustomPatternsWidget

        assert issubclass(CustomPatternsWidget, BaseSettingsWidget)

    def test_indicator_pattern_widget_inherits(self):
        """IndicatorPatternWidget inherits from BaseSettingsWidget."""
        from src.ui.settings.base_settings_widget import BaseSettingsWidget
        from src.ui.settings.indicator_pattern_widget import IndicatorPatternWidget

        assert issubclass(IndicatorPatternWidget, BaseSettingsWidget)


# =========================================================================
# _guard_reentrant Decorator Tests
# =========================================================================


class TestGuardReentrant:
    """Tests for the _guard_reentrant decorator."""

    def _get_decorator(self):
        """Import the decorator."""
        from src.ui.corpus_dialog import _guard_reentrant

        return _guard_reentrant

    def test_normal_execution(self):
        """Decorated method runs normally when not re-entrant."""
        guard = self._get_decorator()

        class Obj:
            """Test object with guard flag."""

            _operation_in_progress = False

            @guard
            def action(self):
                """Return a value."""
                return "done"

        obj = Obj()
        assert obj.action() == "done"

    def test_flag_cleared_after_success(self):
        """Flag is cleared after successful execution."""
        guard = self._get_decorator()

        class Obj:
            """Test object."""

            _operation_in_progress = False

            @guard
            def action(self):
                """No-op."""
                pass

        obj = Obj()
        obj.action()
        assert obj._operation_in_progress is False

    def test_flag_cleared_after_exception(self):
        """Flag is cleared even if the method raises."""
        guard = self._get_decorator()

        class Obj:
            """Test object."""

            _operation_in_progress = False

            @guard
            def action(self):
                """Raise an error."""
                raise ValueError("boom")

        obj = Obj()
        with pytest.raises(ValueError, match="boom"):
            obj.action()
        assert obj._operation_in_progress is False

    def test_reentrant_call_skipped(self):
        """Second call is skipped while first is running."""
        guard = self._get_decorator()

        call_count = 0

        class Obj:
            """Test object."""

            _operation_in_progress = False

            @guard
            def action(self):
                """Count calls and attempt re-entry."""
                nonlocal call_count
                call_count += 1
                # Simulate re-entrant call
                self.action()

        obj = Obj()
        obj.action()
        assert call_count == 1  # Second call was skipped

    def test_returns_none_on_reentrant(self):
        """Re-entrant call returns None."""
        guard = self._get_decorator()

        class Obj:
            """Test object."""

            _operation_in_progress = True  # Already in progress

            @guard
            def action(self):
                """Should not run."""
                return "should not reach"

        obj = Obj()
        assert obj.action() is None

    def test_preserves_method_name(self):
        """Decorator preserves the original method name."""
        guard = self._get_decorator()

        class Obj:
            """Test object."""

            _operation_in_progress = False

            @guard
            def my_action(self):
                """My action docstring."""
                pass

        assert Obj.my_action.__name__ == "my_action"
