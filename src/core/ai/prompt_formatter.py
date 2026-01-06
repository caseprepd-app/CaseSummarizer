"""
Prompt Formatter for Ollama Models

Handles model-specific prompt formatting for different Ollama models.
Supports automatic format detection and wrapping for compatibility.
"""

from src.core.utils.logger import debug


def wrap_prompt_for_model(model_name: str, prompt: str) -> str:
    """
    Wrap prompt in model-specific instruction format for compatibility.

    Different Ollama models expect different prompt formats:
    - Llama/Mistral: [INST]...[/INST]
    - Gemma: Raw prompt (no wrapping needed)
    - Neural-Chat: ### User / ### Assistant format
    - Dolphin: ### User / ### Assistant format
    - Qwen: [INST]...[/INST] (Llama-compatible)
    - Others: Raw prompt (fallback for unknown models)

    Args:
        model_name: Model identifier (e.g., "llama2:7b", "mistral:7b")
        prompt: The instruction/summary prompt

    Returns:
        str: Prompt wrapped in model-specific format
    """
    # Extract base model name (everything before the colon)
    base_model = model_name.split(":")[0].lower() if ":" in model_name else model_name.lower()

    debug(f"[PROMPT FORMAT] Detected model type: {base_model} (full name: {model_name})")

    # Apply model-specific wrapping
    if any(x in base_model for x in ["llama", "mistral"]):
        wrapped = f"[INST] {prompt} [/INST]"
        debug("[PROMPT FORMAT] Applied Llama/Mistral format (wrapped with [INST]...[/INST])")
        return wrapped

    elif "gemma" in base_model:
        debug("[PROMPT FORMAT] Applied Gemma format (raw prompt, no wrapping)")
        return prompt

    elif any(x in base_model for x in ["neural-chat", "dolphin"]):
        wrapped = f"### User:\n{prompt}\n\n### Assistant:"
        debug("[PROMPT FORMAT] Applied Neural-Chat/Dolphin format")
        return wrapped

    elif "qwen" in base_model:
        # Qwen models use a similar format to Llama
        wrapped = f"[INST] {prompt} [/INST]"
        debug("[PROMPT FORMAT] Applied Qwen format (Llama-compatible)")
        return wrapped

    else:
        # Default: raw prompt (fallback for unknown/future models)
        debug(
            "[PROMPT FORMAT] Unknown model type, using raw prompt (will work with instruction-tuned models)"
        )
        return prompt
