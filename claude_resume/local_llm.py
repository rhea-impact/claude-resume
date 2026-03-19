"""Local LLM inference via MLX (Apple Silicon) — Gemma 2 2B.

Singleton model loader: first call downloads + loads the model (~1.5GB),
subsequent calls reuse the in-memory model. Designed for fast summarization
tasks where network latency to an API is unacceptable.

Usage:
    from .local_llm import generate
    text = generate("Summarize this conversation...")

The model stays resident in memory after first load. On M4 36GB this uses
~1.5GB of unified memory and generates at ~80-120 tok/s.

Also usable by the session daemon for background indexing.
"""

import os

_model = None
_tokenizer = None

MODEL_ID = "mlx-community/gemma-2-2b-it-4bit"


def _ensure_loaded():
    """Load model on first call, reuse thereafter."""
    global _model, _tokenizer
    if _model is not None:
        return

    from mlx_lm import load
    _model, _tokenizer = load(MODEL_ID)


def generate(prompt: str, max_tokens: int = 200) -> str:
    """Generate text from a prompt using the local Gemma 2B model.

    Returns the generated text (no special tokens, no prompt echo).
    """
    _ensure_loaded()
    from mlx_lm import generate as mlx_generate

    response = mlx_generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response.strip()


def is_available() -> bool:
    """Check if MLX is available (Apple Silicon + mlx-lm installed)."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False
