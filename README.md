# SandLex - AI Contract Negotiation Competition V1

A platform for studying how Large Language Models negotiate contracts through structured, turn-based interactions.

## Overview

SandLex V1 is a minimal implementation (~600 lines) that enables LLMs to negotiate contracts using a tool-based action system. Models engage in realistic back-and-forth negotiation with budget constraints and fixed turn limits.

## Quick Start

### Test Mode (2 models)
```bash
python main.py --test
```

### Full Tournament (8 models)
```bash
python main.py
```

### Custom Configuration
```bash
python main.py --models "mistralai/Mistral-7B-Instruct-v0.3" "microsoft/Phi-3-mini-4k-instruct" --turns 4 --budget 5
```

## Project Structure

```
sandlex/
├── main.py                          # Entry point
├── config.py                        # Configuration settings
├── contracts.py                     # ContractDocument, Article classes
├── agents.py                        # LLMAgent implementation
├── negotiation.py                   # NegotiationSession, TurnManager
├── tournament.py                    # Round-robin tournament logic
├── templates/
│   └── baseline_contract.json       # Starting contract template
├── results/
│   ├── contracts/                   # Final negotiated contracts
│   ├── negotiations/                # Turn-by-turn action logs
│   └── tournaments/                 # Tournament results (CSV)
├── archive/                         # Old implementation (preserved)
│   ├── old_main.py
│   ├── old_lawyer.py
│   └── old_results/
└── CONTRACT_COMPETITION_REDESIGN.md # Detailed design document
```

## How It Works

### Negotiation Protocol

1. **Fixed Turns**: Each negotiation runs for exactly 6 turns (3 per agent)
2. **Alternating Play**: Buyer → Seller → Buyer → Seller → Buyer → Seller
3. **Budget Constraint**: Each agent has 10 edit points to spend across all turns
4. **Tool-Based Actions**: Agents select from predefined actions (not free-form generation)

### Available Actions

| Action | Cost | Description |
|--------|------|-------------|
| EDIT_ARTICLE | 1 | Modify an existing contract article |
| ADD_ARTICLE | 2 | Add a new article to the contract |
| REMOVE_ARTICLE | 2 | Remove an existing article |
| PASS | 0 | Make no changes (signal satisfaction) |

### Contract Structure

Contracts are JSON documents with 7 standard articles:
- `price_terms` - Pricing and total cost
- `delivery_terms` - Delivery location and timeline
- `payment_terms` - Payment schedule and method
- `warranty_terms` - Warranty period and coverage
- `termination_clauses` - Termination conditions
- `dispute_resolution` - How disputes are resolved
- `governing_law` - Governing jurisdiction

## Configuration

Edit `config.py` to customize:

```python
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    # ... add more models
]

NEGOTIATION_CONFIG = {
    "total_turns": 6,
    "budget_per_agent": 10,
}
```

## Output Files

### Negotiation Logs (`results/negotiations/`)
Turn-by-turn JSON logs showing:
- Each action taken
- Budget spent
- Agent reasoning
- Timestamps

### Final Contracts (`results/contracts/`)
JSON files with:
- All articles in final state
- Modification history
- Who changed what

### Tournament Results (`results/tournaments/`)
CSV files with:
- Match pairings
- Role assignments (buyer/seller)
- Budget usage statistics
- File references

## Models Supported

Default configuration includes 8 models:
1. Mistral 7B Instruct
2. Phi-3 Mini 4k
3. Falcon 7B
4. Llama 3 8B
5. Llama 2 7B
6. Gemma 7B
7. Saul 7B (Legal specialist)
8. AdaptLLM Law-Chat

## Requirements

```bash
pip install -r requirements.txt
```

Or use the conda environment:
```bash
conda env create -f llm_310_environment.yml
conda activate llm_310
```

## Design Philosophy

### V1 Focuses On:
- ✅ Simple, understandable code (~600 lines total)
- ✅ Realistic negotiation mechanics
- ✅ Tool-based LLM interaction (JSON actions)
- ✅ Moderate logging for analysis
- ✅ Working foundation for iteration

### V1 Defers:
- ❌ Judge panels (manual contract review instead)
- ❌ Abstract interfaces (single concrete implementation)
- ❌ Sophisticated error handling (basic try/catch)
- ❌ Statistical analysis pipeline
- ❌ Multi-dimensional evaluation

See `CONTRACT_COMPETITION_REDESIGN.md` for full design rationale and future enhancements.

## Example Workflow

```python
# Load baseline contract
contract = ContractDocument.from_json_file("templates/baseline_contract.json")

# Create agents
buyer = LLMAgent("mistralai/Mistral-7B-Instruct-v0.3", role="buyer")
seller = LLMAgent("microsoft/Phi-3-mini-4k-instruct", role="seller")

# Setup negotiation
turn_manager = TurnManager(total_turns=6, budget_per_agent=10)
session = NegotiationSession(buyer, seller, contract, turn_manager)

# Run negotiation
results = session.run()

# Results include:
# - results["final_contract"] - negotiated contract
# - results["action_log"] - all actions taken
# - results["buyer_final_budget"] - unused budget
```

## Troubleshooting

### GPU Memory Issues
Reduce number of concurrent models or use smaller models:
```python
# In config.py
MODELS = MODELS[:4]  # Only use first 4 models
```

### JSON Parsing Errors
Models sometimes fail to generate valid JSON. The system automatically falls back to PASS action. Check negotiation logs to see frequency.

### Model Loading Failures
Some models require authentication or special access:
```bash
huggingface-cli login
```

## Next Steps

After running V1 experiments:
1. Analyze negotiation logs to identify patterns
2. Review contracts for quality and strategic behavior
3. Identify pain points (parsing errors, budget usage, etc.)
4. Iterate: Add features from the appendix as needed

## Citation

If you use SandLex in research:
```
@software{sandlex2025,
  title={SandLex: AI Contract Negotiation Competition},
  author={Your Name},
  year={2025},
  version={1.0}
}
```

## License

[Add your license here]

## Contact

[Add contact information]
