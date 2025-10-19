"""
Configuration settings for the negotiation competition.

This module centralizes all configuration parameters including
model lists, negotiation settings, and file paths.
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model Configuration
# List of HuggingFace model identifiers to use in tournament

# Test models (small, fast)
TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "google/gemma-3-270m"
]

# Full tournament models
FULL_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "google/gemma-7b",
    "Equall/Saul-7B-Instruct-v1",
    "AdaptLLM/law-chat"
]

# Default to full models (main.py will override in test mode)
MODELS = FULL_MODELS

# Negotiation Parameters
NEGOTIATION_CONFIG = {
    "total_turns": 6,           # Fixed number of turns per negotiation
    "budget_per_agent": 10,     # Edit budget for each agent
    "turn_order": ["buyer", "seller"] * 3  # Alternating turns
}

# File Paths
PATHS = {
    "baseline_contract": BASE_DIR / "templates" / "baseline_contract.json",
    "results_dir": BASE_DIR / "results",
    "contracts_dir": BASE_DIR / "results" / "contracts",
    "negotiations_dir": BASE_DIR / "results" / "negotiations",
    "tournaments_dir": BASE_DIR / "results" / "tournaments"
}

# Ensure directories exist
for path_key, path_value in PATHS.items():
    if path_key.endswith("_dir"):
        path_value.mkdir(parents=True, exist_ok=True)

# Tournament Settings
TOURNAMENT_CONFIG = {
    "save_intermediate_results": True,  # Save after each match
    "verbose": True,  # Print detailed progress
}

# Model-specific settings (if needed for fine-tuning)
MODEL_SPECIFIC_CONFIG = {
    # Example: different max_tokens or temperature for specific models
    # "mistralai/Mistral-7B-Instruct-v0.3": {"max_new_tokens": 600},
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model."""
    return MODEL_SPECIFIC_CONFIG.get(model_name, {})


def print_config() -> None:
    """Print current configuration."""
    print("="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"\nModels ({len(MODELS)}):")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")

    print(f"\nNegotiation Settings:")
    for key, value in NEGOTIATION_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nFile Paths:")
    for key, value in PATHS.items():
        print(f"  {key}: {value}")

    print("="*70 + "\n")
