"""
Prompt Template Manager for LocalScribe
Manages model-specific prompt templates with validation and preset discovery.

Supports dual-directory system:
- Built-in prompts in config/prompts/ (shipped with app)
- User prompts in %APPDATA%/LocalScribe/prompts/ (persist through updates)

Moved from src/prompt_template_manager.py to src/prompting/template_manager.py in Session 33.
"""

from pathlib import Path

# Skeleton template filename (underscore prefix = excluded from dropdown)
SKELETON_FILENAME = "_template.txt"
README_FILENAME = "_README.txt"

# Skeleton template for users to customize (created in user prompts dir on first run)
USER_SKELETON_TEMPLATE = """<|system|>
You are a legal document summarizer. Customize this template to change how summaries are generated.
<|end|>
<|user|>
Summarize the following legal document.

WORD COUNT: Between {min_words} and {max_words_range} words (target: {max_words} words)

YOUR CUSTOM INSTRUCTIONS HERE:
- Add your own focus areas
- Specify output format preferences
- Include domain-specific guidance

DOCUMENT:
---
{case_text}
---
<|end|>
<|assistant|>
"""

# README content explaining how to create custom prompts
USER_README_CONTENT = """# Creating Custom Prompts for LocalScribe

This folder contains prompt templates that control how LocalScribe summarizes your documents.
Any .txt file you place here (except files starting with underscore _) will appear in the
"Prompt Style" dropdown in the application.

## Quick Start

1. Copy `_template.txt` and rename it (e.g., `my-custom-prompt.txt`)
2. Edit the file to customize the summarization instructions
3. Restart LocalScribe - your prompt will appear in the dropdown

## Required Format

Your prompt MUST include these Phi-3 chat tokens in this order:

    <|system|>
    Your system instructions here (who the AI should act as)
    <|end|>
    <|user|>
    Your detailed instructions here
    <|end|>
    <|assistant|>

## Required Variables

Include these placeholders - LocalScribe will replace them automatically:

    {min_words}      - Minimum word count (e.g., 180)
    {max_words}      - Target word count (e.g., 200)
    {max_words_range} - Maximum word count (e.g., 220)
    {case_text}      - The actual document text to summarize

## Tips for Effective Prompts

1. **Be Specific**: Instead of "summarize this", say exactly what to include:
   - Key parties and their roles
   - Important dates and timeline
   - Legal claims or charges
   - Outcomes or current status

2. **Set the Tone**: Tell the AI what style to use:
   - "Write in plain language for non-lawyers"
   - "Use formal legal terminology"
   - "Focus on actionable information"

3. **Specify What to Avoid**:
   - "Do NOT speculate or interpret motives"
   - "Do NOT include section headers in the output"
   - "Avoid redundant phrases like 'This document...'"

4. **Structure the Output**: Request a specific format:
   - "Present information chronologically"
   - "Use bullet points for key facts"
   - "Start with the most important finding"

5. **Word Count Guidance**: The AI doesn't count words precisely, so:
   - Emphasize the target: "approximately {max_words} words"
   - Give a range: "between {min_words} and {max_words_range} words"

## Example Prompts

See the built-in prompts in the application's config/prompts folder for examples:
- factual-summary.txt - Objective, fact-focused summaries
- strategic-analysis.txt - Analysis with legal strategy insights

## Troubleshooting

If your prompt doesn't appear in the dropdown:
- Make sure the filename ends in .txt
- Make sure the filename does NOT start with underscore (_)
- Restart LocalScribe after adding the file

If you get validation errors:
- Check that all four Phi-3 tokens are present
- Check that all four variables are present
- Make sure <|system|> comes before <|user|> comes before <|assistant|>
"""

# Generic fallback prompt template (based on factual-summary)
# Used when no prompts exist for a model
GENERIC_FALLBACK_TEMPLATE = """<|system|>
You are a legal document summarizer. Your role is to provide clear, objective, factual summaries of legal documents without interpretation or speculation.
<|end|>
<|user|>
Summarize the following legal document with these requirements:

WORD COUNT: Between {min_words} and {max_words_range} words (target: {max_words} words)

FOCUS ON FACTS:
- Identify all parties involved (plaintiffs, defendants, attorneys, judges)
- State the claims, charges, or legal issues presented
- Report procedural status and timeline of events
- List key facts and evidence mentioned
- Note any rulings, decisions, or outcomes

OUTPUT STYLE:
- Write in clear, plain language
- Present facts objectively without analysis
- Use active voice and affirmative statements
- Structure information chronologically when possible
- Do NOT speculate, interpret motives, or analyze strategy
- Do NOT include labels, headers, or meta-commentary in your response

DOCUMENT:
---
{case_text}
---
<|end|>
<|assistant|>
"""


class PromptTemplateManager:
    """
    Manages prompt templates for different AI models.

    Supports a dual-directory system:
    - Built-in prompts in prompts_base_dir (config/prompts/{model_name}/)
    - User prompts in user_prompts_dir (%APPDATA%/LocalScribe/prompts/{model_name}/)

    User prompts override built-in prompts with the same name.
    Each template file can use variables: {min_words}, {max_words},
    {max_words_range}, {case_text}
    """

    def __init__(self, prompts_base_dir: Path, user_prompts_dir: Path = None):
        """
        Initialize the template manager.

        Args:
            prompts_base_dir: Base directory for built-in prompts (config/prompts/)
            user_prompts_dir: Base directory for user prompts (%APPDATA%/LocalScribe/prompts/)
                             If None, only built-in prompts are used.
        """
        self.prompts_base_dir = Path(prompts_base_dir)
        self.user_prompts_dir = Path(user_prompts_dir) if user_prompts_dir else None
        self._cache = {}  # Template cache to avoid repeated file reads

    def get_available_models(self) -> list[str]:
        """
        Get list of models that have prompt templates.

        Returns:
            List of model names (directory names in prompts_base_dir)
        """
        if not self.prompts_base_dir.exists():
            return []

        return [
            d.name for d in self.prompts_base_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

    def get_available_presets(self, model_name: str) -> list[dict[str, str]]:
        """
        Get available prompt presets for a specific model from both directories.

        Merges built-in and user prompts. User prompts with the same ID override
        built-in ones, allowing users to customize default behavior.

        Files starting with underscore (_) are excluded from the list - these are
        reserved for special files like _template.txt and _README.txt.

        Args:
            model_name: Name of the model (e.g., 'phi-3-mini')

        Returns:
            List of dicts with keys: 'name', 'id', 'file_path', 'source'
            Source is 'built-in' or 'custom' for tooltip display
        """
        presets_by_id = {}  # Use dict to handle overrides

        # 1. Load built-in prompts first
        builtin_dir = self.prompts_base_dir / model_name
        if builtin_dir.exists():
            for template_file in builtin_dir.glob('*.txt'):
                # Skip underscore-prefixed files (reserved for _template.txt, _README.txt)
                if template_file.name.startswith('_'):
                    continue
                preset_id = template_file.stem
                display_name = preset_id.replace('-', ' ').replace('_', ' ').title()
                presets_by_id[preset_id] = {
                    'name': display_name,
                    'id': preset_id,
                    'file_path': str(template_file),
                    'source': 'built-in'
                }

        # 2. Load user prompts (override built-in if same ID)
        if self.user_prompts_dir:
            user_dir = self.user_prompts_dir / model_name
            if user_dir.exists():
                for template_file in user_dir.glob('*.txt'):
                    # Skip underscore-prefixed files (reserved for _template.txt, _README.txt)
                    if template_file.name.startswith('_'):
                        continue
                    preset_id = template_file.stem
                    display_name = preset_id.replace('-', ' ').replace('_', ' ').title()
                    presets_by_id[preset_id] = {
                        'name': display_name,
                        'id': preset_id,
                        'file_path': str(template_file),
                        'source': 'custom'
                    }

        return sorted(presets_by_id.values(), key=lambda x: x['name'])

    def load_template(
        self,
        model_name: str,
        preset_id: str,
        use_cache: bool = True
    ) -> str:
        """
        Load a prompt template from file.

        Checks user prompts directory first (if configured), then built-in directory.
        This allows user prompts to override built-in ones.

        Args:
            model_name: Name of the model (e.g., 'phi-3-mini')
            preset_id: Template identifier (filename without extension)
            use_cache: Whether to use cached template if available

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template file doesn't exist in either directory
            ValueError: If template is invalid
        """
        cache_key = f"{model_name}/{preset_id}"

        # Check cache first
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Try to find template (user directory first, then built-in)
        template_path = None

        # 1. Check user prompts directory first (higher priority)
        if self.user_prompts_dir:
            user_path = self.user_prompts_dir / model_name / f"{preset_id}.txt"
            if user_path.exists():
                template_path = user_path

        # 2. Fall back to built-in directory
        if template_path is None:
            builtin_path = self.prompts_base_dir / model_name / f"{preset_id}.txt"
            if builtin_path.exists():
                template_path = builtin_path

        if template_path is None:
            raise FileNotFoundError(
                f"Template '{preset_id}' not found for model '{model_name}'\n"
                f"Searched: {self.prompts_base_dir / model_name}, "
                f"{self.user_prompts_dir / model_name if self.user_prompts_dir else 'N/A'}"
            )

        with open(template_path, encoding='utf-8') as f:
            template = f.read()

        # Validate template
        self.validate_template(template, template_path)

        # Cache and return
        self._cache[cache_key] = template
        return template

    def validate_template(self, template: str, template_path: Path = None) -> None:
        """
        Validate that a template has required elements.

        Args:
            template: Template content to validate
            template_path: Optional path for better error messages

        Raises:
            ValueError: If template is missing required elements
        """
        errors = []

        # Check for required Phi-3 chat tokens
        required_tokens = ['<|system|>', '<|user|>', '<|end|>', '<|assistant|>']
        for token in required_tokens:
            if token not in template:
                errors.append(f"Missing required token: {token}")

        # Check for required template variables
        required_vars = ['{min_words}', '{max_words}', '{max_words_range}', '{case_text}']
        for var in required_vars:
            if var not in template:
                errors.append(f"Missing required variable: {var}")

        # Check template structure
        if template.find('<|system|>') > template.find('<|user|>'):
            errors.append("<|system|> must come before <|user|>")

        if template.find('<|user|>') > template.find('<|assistant|>'):
            errors.append("<|user|> must come before <|assistant|>")

        # Raise error if validation failed
        if errors:
            location = f" in {template_path}" if template_path else ""
            raise ValueError(
                f"Template validation failed{location}:\n" +
                "\n".join(f"  - {error}" for error in errors)
            )

    def format_template(
        self,
        template: str,
        min_words: int,
        max_words: int,
        max_words_range: int,
        case_text: str
    ) -> str:
        """
        Format a template with provided values.

        Args:
            template: Template string with {variables}
            min_words: Minimum word count
            max_words: Target word count
            max_words_range: Maximum word count
            case_text: Legal document text to summarize

        Returns:
            Formatted prompt ready for model input
        """
        return template.format(
            min_words=min_words,
            max_words=max_words,
            max_words_range=max_words_range,
            case_text=case_text
        )

    def get_default_preset_id(self, model_name: str) -> str | None:
        """
        Get the default preset for a model (first alphabetically).

        Args:
            model_name: Name of the model

        Returns:
            Preset ID string, or None if no presets available
        """
        presets = self.get_available_presets(model_name)
        if not presets:
            return None
        return presets[0]['id']

    def clear_cache(self):
        """Clear the template cache."""
        self._cache.clear()

    def ensure_generic_fallback(self, model_name: str) -> None:
        """
        Ensure a generic fallback prompt exists for a model.

        If the model directory has no prompts, creates a 'generic-summary.txt'
        file with the standard fallback template.

        Args:
            model_name: Name of the model (e.g., 'phi-3-mini')
        """
        model_dir = self.prompts_base_dir / model_name

        # Check if any prompts exist
        presets = self.get_available_presets(model_name)

        if not presets:
            # No prompts found - create generic fallback
            model_dir.mkdir(parents=True, exist_ok=True)
            generic_file = model_dir / "generic-summary.txt"

            if not generic_file.exists():
                with open(generic_file, 'w', encoding='utf-8') as f:
                    f.write(GENERIC_FALLBACK_TEMPLATE)

                print(f"Created generic fallback prompt for {model_name}")

    def get_best_default_preset(
        self,
        model_name: str,
        user_preference: str | None = None
    ) -> str:
        """
        Get the best default preset to use for a model.

        Priority order:
        1. User's saved preference (if exists and file found)
        2. First preset alphabetically
        3. 'generic-summary' if no other presets exist

        Args:
            model_name: Name of the model
            user_preference: User's preferred preset ID (from preferences)

        Returns:
            Preset ID to use as default
        """
        # Ensure generic fallback exists
        self.ensure_generic_fallback(model_name)

        # Get available presets
        presets = self.get_available_presets(model_name)

        if not presets:
            return "generic-summary"  # Shouldn't happen after ensure_generic_fallback

        # Try user preference first
        if user_preference:
            preset_ids = [p['id'] for p in presets]
            if user_preference in preset_ids:
                return user_preference

        # Fall back to first preset alphabetically
        return presets[0]['id']

    def ensure_user_skeleton(self, model_name: str) -> Path | None:
        """
        Ensure skeleton template and README exist in the user prompts directory.

        Creates the user prompts directory and two helper files if they don't exist:
        - _template.txt: Starting point for creating custom prompts
        - _README.txt: Instructions for creating and using custom prompts

        Both files start with underscore (_) so they won't appear in the dropdown.

        Args:
            model_name: Name of the model (e.g., 'phi-3-mini')

        Returns:
            Path to the user prompts directory, or None if user_prompts_dir not configured
        """
        if not self.user_prompts_dir:
            return None

        user_model_dir = self.user_prompts_dir / model_name
        user_model_dir.mkdir(parents=True, exist_ok=True)

        # Create skeleton template (underscore prefix = hidden from dropdown)
        skeleton_file = user_model_dir / SKELETON_FILENAME
        if not skeleton_file.exists():
            with open(skeleton_file, 'w', encoding='utf-8') as f:
                f.write(USER_SKELETON_TEMPLATE)

        # Create README with instructions (underscore prefix = hidden from dropdown)
        readme_file = user_model_dir / README_FILENAME
        if not readme_file.exists():
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(USER_README_CONTENT)

        return user_model_dir

    def get_user_prompts_path(self, model_name: str) -> Path | None:
        """
        Get the path to the user prompts directory for a model.

        Useful for displaying in tooltips to guide users where to add custom prompts.

        Args:
            model_name: Name of the model

        Returns:
            Path to user prompts directory, or None if not configured
        """
        if not self.user_prompts_dir:
            return None
        return self.user_prompts_dir / model_name
