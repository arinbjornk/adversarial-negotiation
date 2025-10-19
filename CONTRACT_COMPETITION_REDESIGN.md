# Contract Competition System - Redesign Plan

## Project Overview

**Goal**: Transform the existing AI contract competition system into a scientifically rigorous experimental platform for studying AI negotiation capabilities.

**Current System Problems**:
- Unstructured contract editing leading to inconsistent results (agents rewrite entire contracts from scratch each turn)
- Poor judgment quality with corrupted responses
- No change tracking between negotiation turns
- Binary evaluation inadequate for contract complexity
- Lack of experimental controls and reproducibility
- No constraints preventing agents from making unlimited changes per turn

**Redesign Goals**:
- **Realistic negotiation**: Free-form legal writing within controlled structural constraints
- **Experimental control**: Fixed turn limits and budget systems for reproducible experiments
- **Incremental changes**: Article-level modifications instead of complete rewrites
- **Transparent tracking**: Visible change history and diff visualization
- **Comparable results**: All negotiations run through identical turn structure

**Implementation Philosophy**:
- **Start simple**: Build minimal working system first (~500 lines, 4-6 weeks)
- **Iterate based on real problems**: Add complexity only when needed
- **Keep it verifiable**: Prioritize working code over elegant architecture
- **Future-ready**: Detailed design in appendix for future enhancements

## Minimal Viable Implementation (Phase 1)

### Quick Start Goal
Get experimental results in 4-6 weeks with minimal but functional system.

### Core Components (Simple Version)

```
sandlex/
├── contracts.py          # ContractDocument class, basic JSON validation
├── agents.py             # LLMAgent (one simple implementation)
├── negotiation.py        # NegotiationSession, TurnManager
├── tournament.py         # Round-robin tournament runner
├── config.py             # Experiment configuration
└── main.py              # Entry point
```

### Implementation Checklist

**Week 1-2: Core Contract System**
- [ ] ContractDocument class with JSON articles
- [ ] Basic JSON schema validation
- [ ] Action classes (EditArticle, AddArticle, RemoveArticle, Pass)
- [ ] Simple contract.apply_action() method

**Week 2-3: LLM Agent**
- [ ] LLMAgent with simple JSON prompt
- [ ] Basic JSON parsing (try/catch with fallback to PASS)
- [ ] GPU assignment (cycle through available GPUs)
- [ ] Test with 1-2 models

**Week 3-4: Negotiation Engine**
- [ ] TurnManager (6 fixed turns, budget tracking)
- [ ] NegotiationSession.run() - simple loop
- [ ] Basic change tracking (store history list)
- [ ] CSV logging of actions

**Week 4-5: Tournament System**
- [ ] Round-robin match pairing
- [ ] Role randomization (buyer/seller)
- [ ] Results collection
- [ ] CSV export

**Week 5-6: Evaluation & Testing**
- [ ] Run full 8-model tournament
- [ ] Analyze results
- [ ] Identify what needs improvement
- [ ] Document pain points

### Simplified Code Sketches

#### contracts.py (~100 lines)
```python
import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Article:
    content: str
    last_modified_by: str

class ContractDocument:
    def __init__(self, articles: Dict[str, Article]):
        self.articles = articles
        self.history: List[dict] = []

    def to_dict(self) -> dict:
        return {
            "articles": {
                name: {"content": art.content, "last_modified_by": art.last_modified_by}
                for name, art in self.articles.items()
            }
        }

    def apply_action(self, action, agent_id: str):
        if action["action_type"] == "EDIT_ARTICLE":
            self.articles[action["article_name"]] = Article(
                content=action["content"],
                last_modified_by=agent_id
            )
        elif action["action_type"] == "ADD_ARTICLE":
            self.articles[action["article_name"]] = Article(
                content=action["content"],
                last_modified_by=agent_id
            )
        elif action["action_type"] == "REMOVE_ARTICLE":
            del self.articles[action["article_name"]]
        # PASS does nothing

        self.history.append({"agent": agent_id, "action": action})
```

#### agents.py (~150 lines)
```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMAgent:
    def __init__(self, model_name: str, role: str, gpu_id: int):
        self.model_name = model_name
        self.role = role
        self.device = f"cuda:{gpu_id}"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def select_action(self, contract: ContractDocument, budget: int, turns: int) -> dict:
        prompt = f"""You are negotiating a contract as the {self.role}.

CONTRACT:
{json.dumps(contract.to_dict(), indent=2)}

BUDGET: {budget} edits remaining
TURNS: {turns} turns remaining

Respond with JSON:
{{
  "action_type": "EDIT_ARTICLE" | "ADD_ARTICLE" | "REMOVE_ARTICLE" | "PASS",
  "article_name": "price_terms",
  "content": "The new article text...",
  "reasoning": "Why you're doing this"
}}

Your JSON:
"""

        response = self._generate(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback on parse error
            return {"action_type": "PASS", "reasoning": "Parse error"}

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=500)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### negotiation.py (~100 lines)
```python
class TurnManager:
    def __init__(self, total_turns: int = 6, budget_per_agent: int = 10):
        self.total_turns = total_turns
        self.budget_per_agent = budget_per_agent
        self.budgets = {}
        self.current_turn = 0

    def is_complete(self) -> bool:
        return self.current_turn >= self.total_turns

    def deduct_budget(self, agent_id: str, cost: int):
        self.budgets[agent_id] -= cost

class NegotiationSession:
    def __init__(self, buyer, seller, initial_contract, turn_manager):
        self.buyer = buyer
        self.seller = seller
        self.contract = initial_contract
        self.turn_manager = turn_manager
        self.agents = {"buyer": buyer, "seller": seller}
        self.turn_order = ["buyer", "seller"] * 3  # 6 turns total

    def run(self):
        for agent_role in self.turn_order:
            agent = self.agents[agent_role]
            budget = self.turn_manager.budgets[agent_role]
            turns_left = self.total_turns - self.turn_manager.current_turn

            action = agent.select_action(self.contract, budget, turns_left)
            self.contract.apply_action(action, agent_role)

            cost = self._get_action_cost(action)
            self.turn_manager.deduct_budget(agent_role, cost)
            self.turn_manager.current_turn += 1

        return self.contract
```

### What We're NOT Building (Yet)

- ❌ Abstract interfaces (LawyerAgent ABC)
- ❌ Multiple LLM interaction strategies
- ❌ Sophisticated error handling with retries
- ❌ Constrained generation libraries
- ❌ Comprehensive logging infrastructure
- ❌ Multi-dimensional evaluation framework
- ❌ Statistical analysis pipeline

**These go in the appendix as "Future Enhancements"**

### Success Criteria for Phase 1

✅ Can run 8 models in round-robin tournament
✅ Contracts are modified incrementally (not rewritten)
✅ Budget system prevents runaway edits
✅ Results saved to CSV for analysis
✅ Can identify which models negotiate better
✅ System is understandable and debuggable

### When to Add Complexity

Only add from appendix when you hit a **real problem**:
- JSON parsing fails >20% of the time → Add retry logic
- Need to compare with GPT-4 → Add abstract interface
- Hard to debug issues → Add structured logging
- Results not reproducible → Add seed management

## Core Design Decisions

### Turn-Based Protocol
- **Fixed turns**: Exactly 6 turns per negotiation (3 per agent)
- **Alternating play**: Buyer → Seller → Buyer → Seller → Buyer → Seller
- **Edit budget**: 10 edits per agent across all turns
- **No early termination**: Always runs full turn cycle for experimental consistency

### Action System
| Action | Cost | Description |
|--------|------|-------------|
| EDIT_ARTICLE | 1 | Modify existing article with free-form text |
| ADD_ARTICLE | 2 | Add new article to contract |
| REMOVE_ARTICLE | 2 | Remove existing article |
| PASS | 0 | Make no changes (signal satisfaction) |

### Contract Structure
- **Format**: JSON with named articles
- **Articles**: price_terms, delivery_terms, warranty_terms, payment_terms, etc.
- **Metadata**: Tracks who modified what and when
- **Validation**: Basic JSON schema checking

### LLM Interaction
- **Input**: Contract state as JSON + role/budget/turns info
- **Output**: JSON action object
- **Fallback**: PASS on parse errors
- **One strategy**: Simple prompt engineering with JSON response

---

## Detailed Design Reference (For Context Only)

The sections below provide context for design decisions. They are NOT required for the minimal implementation but help understand the "why" behind choices.

### Contract JSON Schema Example
│  ┌──────────────────────┐        ┌─────────────────────┐        │
│  │ Tournament           │───────►│ Results             │        │
│  │ Orchestrator         │        │ Analyzer            │        │
│  │                      │        │                     │        │
│  │ - Match scheduling   │        │ - Statistical       │        │
│  │ - Round-robin logic  │        │   analysis          │        │
│  │ - Role assignment    │        │ - Report generation │        │
│  └──────────────────────┘        └─────────────────────┘        │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│                       DOMAIN LAYER                               │
│  ┌────────────────────────────────────────────────┐              │
│  │ NegotiationSession                             │              │
│  │ ┌────────────────┐  ┌─────────────────────┐   │              │
│  │ │ TurnManager    │  │ ChangeTracker       │   │              │
│  │ │                │  │                     │   │              │
│  │ │ - Execute turn │  │ - Record diffs      │   │              │
│  │ │ - Track budget │  │ - History tracking  │   │              │
│  │ └────────────────┘  └─────────────────────┘   │              │
│  └────────────────────────────────────────────────┘              │
│                                                                   │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ ContractDocument     │   │ ContractValidator       │         │
│  │                      │   │                         │         │
│  │ - Articles           │   │ - Schema validation     │         │
│  │ - Metadata           │   │ - Business rules        │         │
│  │ - to_dict()          │   │ - Legal completeness    │         │
│  └──────────────────────┘   └─────────────────────────┘         │
│                                                                   │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ LawyerAgent          │   │ JudgePanel              │         │
│  │ (Abstract Interface) │   │ (Abstract Interface)    │         │
│  │                      │   │                         │         │
│  │ + select_action()    │   │ + evaluate_contract()   │         │
│  └──────────────────────┘   └─────────────────────────┘         │
└──────────────┬──────────────────────┬───────────────────────────┘
               │                      │
┌──────────────▼──────────────────────▼───────────────────────────┐
│                   INFRASTRUCTURE LAYER                           │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ LLMLawyerAgent       │   │ LLMJudgePanel           │         │
│  │ (Concrete)           │   │ (Concrete)              │         │
│  │                      │   │                         │         │
│  │ - Build prompts      │   │ - Scoring prompts       │         │
│  │ - Parse JSON actions │   │ - Multi-model ensemble  │         │
│  │ - Handle errors      │   │                         │         │
│  └──────────┬───────────┘   └─────────────────────────┘         │
│             │                                                    │
│  ┌──────────▼───────────┐   ┌─────────────────────────┐         │
│  │ ModelManager         │   │ DataStorage             │         │
│  │                      │   │                         │         │
│  │ - GPU allocation     │   │ - ContractRepository    │         │
│  │ - Model loading      │   │ - ResultsRepository     │         │
│  │ - Memory management  │   │ - JSON/CSV persistence  │         │
│  └──────────────────────┘   └─────────────────────────┘         │
│                                                                   │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ NegotiationLogger    │   │ ConfigurationManager    │         │
│  │                      │   │                         │         │
│  │ - Structured logs    │   │ - Experiment configs    │         │
│  │ - Event tracking     │   │ - Model parameters      │         │
│  └──────────────────────┘   └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Dependency Inversion**: Domain layer defines interfaces (LawyerAgent, JudgePanel), infrastructure layer implements them
2. **Domain Independence**: Core negotiation logic knows nothing about LLMs, GPUs, or file formats
3. **Testability**: Can test domain logic with mock agents (no GPU required)
4. **Swappability**: Easy to switch from HuggingFace models to OpenAI API or rule-based agents
5. **Clear Data Flow**: Application → Domain → Infrastructure (never backwards)

### Module Structure

```
sandlex/
├── domain/                      # Core business logic (no external dependencies)
│   ├── contracts/
│   │   ├── contract_document.py      # ContractDocument, Article classes
│   │   ├── contract_validator.py     # Validation logic
│   │   └── actions.py                # NegotiationAction types
│   ├── negotiation/
│   │   ├── negotiation_session.py    # Orchestrates one negotiation
│   │   ├── turn_manager.py           # Turn execution, budget tracking
│   │   └── change_tracker.py         # Records diffs between turns
│   └── agents/
│       ├── lawyer_agent.py           # Abstract LawyerAgent interface
│       └── judge_panel.py            # Abstract JudgePanel interface
│
├── application/                 # Use cases and orchestration
│   ├── tournament/
│   │   ├── tournament_orchestrator.py
│   │   └── match_scheduler.py
│   └── analysis/
│       └── results_analyzer.py
│
├── infrastructure/              # Technical implementation details
│   ├── llm/
│   │   ├── model_manager.py          # GPU allocation, model loading
│   │   ├── llm_lawyer_agent.py       # Concrete LawyerAgent using LLMs
│   │   ├── llm_judge_panel.py        # Concrete JudgePanel using LLMs
│   │   └── prompt_builder.py         # Prompt engineering utilities
│   ├── storage/
│   │   ├── contract_repository.py    # Save/load contracts
│   │   └── results_repository.py     # Save/load results
│   └── logging/
│       └── negotiation_logger.py     # Structured logging
│
└── config/
    ├── experiment_config.py     # Experiment parameters
    └── model_config.py          # Model-specific settings
```

### Class Structure and Interfaces

```python
# ============================================================================
# DOMAIN LAYER - Core business logic
# ============================================================================

# domain/contracts/contract_document.py
class ContractDocument:
    """Represents a contract with articles and metadata."""
    def __init__(self, metadata: ContractMetadata, articles: Dict[str, Article]):
        self.metadata = metadata
        self.articles = articles
        self.validation_status: Optional[ValidationResult] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        pass

    def apply_action(self, action: NegotiationAction) -> 'ContractDocument':
        """Apply an action and return new contract state."""
        pass

# domain/contracts/actions.py
class NegotiationAction:
    """Base class for all negotiation actions."""
    action_type: str
    cost: int

class EditArticleAction(NegotiationAction):
    article_name: str
    new_content: str
    reasoning: str
    cost = 1

class AddArticleAction(NegotiationAction):
    article_name: str
    content: str
    reasoning: str
    cost = 2

class RemoveArticleAction(NegotiationAction):
    article_name: str
    reasoning: str
    cost = 2

class PassAction(NegotiationAction):
    reasoning: str
    cost = 0

# domain/agents/lawyer_agent.py (ABSTRACT INTERFACE)
from abc import ABC, abstractmethod

class LawyerAgent(ABC):
    """Abstract interface for negotiation agents.

    Domain layer depends on this interface.
    Infrastructure layer provides concrete implementations.
    """

    @abstractmethod
    def select_action(
        self,
        contract: ContractDocument,
        remaining_budget: int,
        remaining_turns: int
    ) -> NegotiationAction:
        """Select the next negotiation action.

        The domain layer calls this method without knowing
        whether the agent is an LLM, rule-based system, or human.
        """
        pass

# domain/negotiation/negotiation_session.py
class NegotiationSession:
    """Orchestrates a single negotiation between two agents."""

    def __init__(
        self,
        buyer: LawyerAgent,  # Abstract interface!
        seller: LawyerAgent,  # Abstract interface!
        initial_contract: ContractDocument,
        turn_manager: TurnManager
    ):
        self.buyer = buyer
        self.seller = seller
        self.contract = initial_contract
        self.turn_manager = turn_manager

    def run(self) -> NegotiationResult:
        """Execute the full negotiation."""
        while not self.turn_manager.is_complete():
            current_agent = self.turn_manager.get_current_agent()
            action = current_agent.select_action(
                self.contract,
                self.turn_manager.get_budget(current_agent),
                self.turn_manager.remaining_turns()
            )
            self.contract = self.contract.apply_action(action)
            self.turn_manager.record_action(action)

        return NegotiationResult(self.contract, self.turn_manager.history)

# domain/negotiation/turn_manager.py
class TurnManager:
    """Manages turn order, budgets, and end conditions."""

    def __init__(self, total_turns: int = 6, budget_per_agent: int = 10):
        self.total_turns = total_turns
        self.budget_per_agent = budget_per_agent
        self.current_turn = 0
        self.budgets: Dict[str, int] = {}
        self.history: List[TurnRecord] = []

    def is_complete(self) -> bool:
        """Check if negotiation should end."""
        return self.current_turn >= self.total_turns

# ============================================================================
# INFRASTRUCTURE LAYER - LLM implementation
# ============================================================================

# infrastructure/llm/llm_lawyer_agent.py
class LLMLawyerAgent(LawyerAgent):
    """Concrete implementation using HuggingFace LLMs."""

    def __init__(self, model, tokenizer, role: str, prompt_builder: PromptBuilder):
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.prompt_builder = prompt_builder

    def select_action(
        self,
        contract: ContractDocument,
        remaining_budget: int,
        remaining_turns: int
    ) -> NegotiationAction:
        """Generate action using LLM with structured output."""

        # 1. Build structured prompt
        prompt = self.prompt_builder.build_negotiation_prompt(
            contract=contract,
            role=self.role,
            budget=remaining_budget,
            turns=remaining_turns
        )

        # 2. Generate response (with JSON constraints if supported)
        response_json = self._generate_structured_response(prompt)

        # 3. Validate and parse into domain object
        action = self._parse_action(response_json)

        return action

    def _generate_structured_response(self, prompt: str) -> dict:
        """Generate JSON response from LLM."""
        # Implementation depends on model capabilities
        pass

    def _parse_action(self, response: dict) -> NegotiationAction:
        """Convert JSON to NegotiationAction domain object."""
        action_type = response["action_type"]

        if action_type == "EDIT_ARTICLE":
            return EditArticleAction(
                article_name=response["article_name"],
                new_content=response["content"],
                reasoning=response.get("reasoning", "")
            )
        elif action_type == "PASS":
            return PassAction(reasoning=response.get("reasoning", ""))
        # ... etc
```

## Data Flows

### 1. Tournament Initialization Flow
```
Tournament Start → Load Base Contracts → Initialize Agent Pairs → 
Create Judge Panels → Begin Negotiation Sessions
```

### 2. Negotiation Turn Flow
```
Agent Receives Contract → Analyzes Changes → Selects Action (EDIT/ADD/REMOVE/PASS) →
Validates Action → Updates Contract → Records Change → Deducts Budget →
Increments Turn Counter → Check End Conditions → Next Turn or End
```

**Fixed Turn Protocol:**
- Every negotiation runs for exactly N turns (typically 6)
- Agents alternate: Buyer → Seller → Buyer → Seller → Buyer → Seller
- Agents can PASS (cost 0) if satisfied with current contract
- Consecutive PASSes indicate implicit agreement
- Negotiation always completes full turn cycle for experimental consistency

### 3. Evaluation Flow
```
Final Contract → Judge Panel → Multi-Dimensional Scoring → 
Statistical Analysis → Result Recording → Tournament Update
```

### 4. Data Storage Flow
```
Real-time Events → Structured Logging → Database Storage → 
Analysis Pipeline → Report Generation
```

## Turn Mechanism

### Turn Structure

```python
class TurnManager:
    def __init__(self, total_turns: int = 6, budget_per_agent: int = 10):
        self.total_turns = total_turns
        self.budget_per_agent = budget_per_agent
        self.current_turn = 0

    def execute_turn(self, agent: LawyerAgent, contract: ContractDocument) -> TurnResult:
        # 1. Present current contract state with changes highlighted
        contract_view = self.prepare_contract_view(contract, agent.role)

        # 2. Get agent action with budget constraints
        remaining_budget = self.get_budget(agent)
        remaining_turns = self.total_turns - self.current_turn
        action = agent.select_action(contract_view, remaining_budget, remaining_turns)

        # 3. Validate and apply action
        if self.validate_action(action, agent):
            updated_contract = self.apply_action(contract, action)
            self.deduct_budget(agent, action.cost)
            self.record_change(action, agent)
            self.current_turn += 1
            return TurnResult.SUCCESS(updated_contract)
        else:
            return TurnResult.INVALID_ACTION(action.validation_error)

    def is_negotiation_complete(self) -> bool:
        """Check if negotiation should end."""
        # Primary end condition: fixed turn limit
        if self.current_turn >= self.total_turns:
            return True

        # Secondary: both agents out of budget
        if all(self.get_budget(agent) == 0 for agent in self.agents):
            return True

        return False
```

**Key Configuration Parameters:**
```python
NEGOTIATION_CONFIG = {
    "total_turns": 6,           # Fixed number of turns (3 per agent in alternating order)
    "edit_budget_per_agent": 10, # Total budget available across all turns
    "turn_order": ["buyer", "seller", "buyer", "seller", "buyer", "seller"],
}
```

### Action Types and Costs

| Action Type | Edit Cost | Description |
|-------------|-----------|-------------|
| EDIT_ARTICLE | 1 | Modify existing article content with free-form text |
| ADD_ARTICLE | 2 | Add new contract article |
| REMOVE_ARTICLE | 2 | Remove existing article |
| PASS | 0 | Make no changes this turn (signal satisfaction) |

**Design Philosophy:**
- **Free-form writing within articles**: Agents write natural legal language, not pre-defined templates
- **Structural constraints**: Agents can only modify one article per action, preventing complete contract rewrites
- **Budget pressure**: Limited edit budget forces strategic prioritization of changes
- **Article-level granularity**: Changes are tracked at the article level, making modifications transparent

This design simulates realistic contract negotiation where lawyers incrementally refine terms rather than regenerating entire contracts each turn.

### Negotiation End Conditions
1. **Fixed turn limit reached** (primary end condition - ensures experimental control)
2. Edit budgets exhausted for both agents
3. Both agents PASS consecutively (optional early termination, configurable)

## Contract Structure

### JSON Schema
```json
{
  "contract": {
    "metadata": {
      "contract_id": "string",
      "created_timestamp": "datetime",
      "parties": {
        "buyer": "West Manufacturing Inc.",
        "seller": "Square Machines Inc."
      },
      "subject": "Sale of 100 machines"
    },
    "articles": {
      "price_terms": {
        "content": "string",
        "last_modified_by": "agent_id",
        "modification_timestamp": "datetime"
      },
      "delivery_terms": { ... },
      "warranty_terms": { ... },
      "payment_terms": { ... },
      "termination_clauses": { ... }
    },
    "validation": {
      "is_complete": "boolean",
      "missing_articles": ["list"],
      "legal_issues": ["list"]
    }
  }
}
```

### Standard Article Types
1. **Price and Payment Terms**
2. **Delivery and Performance**
3. **Warranties and Representations**
4. **Risk Allocation and Liability**
5. **Termination and Default**
6. **Dispute Resolution**
7. **Governing Law**

## JSON Validation Framework

### Schema Definition
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "contract": {
      "type": "object",
      "properties": {
        "metadata": {
          "type": "object",
          "properties": {
            "contract_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]+$"},
            "created_timestamp": {"type": "string", "format": "date-time"},
            "parties": {
              "type": "object",
              "properties": {
                "buyer": {"type": "string", "minLength": 1},
                "seller": {"type": "string", "minLength": 1}
              },
              "required": ["buyer", "seller"]
            },
            "subject": {"type": "string", "minLength": 1}
          },
          "required": ["contract_id", "created_timestamp", "parties", "subject"]
        },
        "articles": {
          "type": "object",
          "properties": {
            "price_terms": {"$ref": "#/definitions/article"},
            "delivery_terms": {"$ref": "#/definitions/article"},
            "warranty_terms": {"$ref": "#/definitions/article"},
            "payment_terms": {"$ref": "#/definitions/article"},
            "termination_clauses": {"$ref": "#/definitions/article"},
            "dispute_resolution": {"$ref": "#/definitions/article"},
            "governing_law": {"$ref": "#/definitions/article"}
          },
          "required": ["price_terms", "delivery_terms"],
          "additionalProperties": {"$ref": "#/definitions/article"}
        },
        "validation": {
          "type": "object",
          "properties": {
            "is_complete": {"type": "boolean"},
            "missing_articles": {
              "type": "array",
              "items": {"type": "string"}
            },
            "legal_issues": {
              "type": "array",
              "items": {"type": "string"}
            },
            "last_validated": {"type": "string", "format": "date-time"}
          },
          "required": ["is_complete", "missing_articles", "legal_issues"]
        }
      },
      "required": ["metadata", "articles", "validation"]
    }
  },
  "definitions": {
    "article": {
      "type": "object",
      "properties": {
        "content": {"type": "string", "minLength": 1},
        "last_modified_by": {"type": "string"},
        "modification_timestamp": {"type": "string", "format": "date-time"},
        "word_count": {"type": "integer", "minimum": 0},
        "complexity_score": {"type": "number", "minimum": 0, "maximum": 1}
      },
      "required": ["content", "last_modified_by", "modification_timestamp"]
    }
  }
}
```

### Validation Implementation
```python
import jsonschema
from jsonschema import validate, ValidationError
from datetime import datetime
import json

class ContractValidator:
    def __init__(self, schema_path: str = None):
        if schema_path:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        else:
            self.schema = self._get_default_schema()
    
    def validate_contract(self, contract_data: dict) -> ValidationResult:
        """Validate contract against JSON schema and business rules."""
        result = ValidationResult()
        
        # 1. JSON Schema Validation
        try:
            validate(instance=contract_data, schema=self.schema)
            result.schema_valid = True
        except ValidationError as e:
            result.schema_valid = False
            result.schema_errors.append(str(e))
        
        # 2. Business Rule Validation
        self._validate_business_rules(contract_data, result)
        
        # 3. Legal Completeness Check
        self._validate_legal_completeness(contract_data, result)
        
        return result
    
    def _validate_business_rules(self, contract: dict, result: ValidationResult):
        """Validate business logic rules."""
        articles = contract.get('contract', {}).get('articles', {})
        
        # Check minimum required articles
        required_articles = ['price_terms', 'delivery_terms']
        for article in required_articles:
            if article not in articles:
                result.business_errors.append(f"Missing required article: {article}")
        
        # Validate article content quality
        for article_name, article_data in articles.items():
            content = article_data.get('content', '')
            if len(content.split()) < 10:  # Minimum 10 words
                result.business_errors.append(f"Article '{article_name}' too short")
    
    def _validate_legal_completeness(self, contract: dict, result: ValidationResult):
        """Check for legal completeness requirements."""
        articles = contract.get('contract', {}).get('articles', {})
        
        # Legal completeness criteria
        legal_requirements = {
            'price_terms': ['price', 'currency', 'payment'],
            'delivery_terms': ['date', 'location', 'method'],
            'warranty_terms': ['warranty', 'duration'],
        }
        
        for article_name, keywords in legal_requirements.items():
            if article_name in articles:
                content = articles[article_name].get('content', '').lower()
                missing_keywords = [kw for kw in keywords if kw not in content]
                if missing_keywords:
                    result.legal_issues.append(
                        f"Article '{article_name}' missing: {', '.join(missing_keywords)}"
                    )

class ValidationResult:
    def __init__(self):
        self.schema_valid: bool = False
        self.schema_errors: List[str] = []
        self.business_errors: List[str] = []
        self.legal_issues: List[str] = []
        self.warnings: List[str] = []
    
    @property
    def is_valid(self) -> bool:
        return (self.schema_valid and 
                len(self.schema_errors) == 0 and 
                len(self.business_errors) == 0)
    
    @property
    def is_legally_complete(self) -> bool:
        return self.is_valid and len(self.legal_issues) == 0
    
    def to_dict(self) -> dict:
        return {
            'schema_valid': self.schema_valid,
            'schema_errors': self.schema_errors,
            'business_errors': self.business_errors,
            'legal_issues': self.legal_issues,
            'warnings': self.warnings,
            'overall_valid': self.is_valid,
            'legally_complete': self.is_legally_complete
        }
```

### Validation Integration Points

#### 1. Contract Creation Validation
```python
def create_contract(base_template: dict) -> ContractDocument:
    validator = ContractValidator()
    validation_result = validator.validate_contract(base_template)
    
    if not validation_result.is_valid:
        raise ContractValidationError(validation_result.schema_errors)
    
    return ContractDocument(base_template, validation_result)
```

#### 2. Action Validation
```python
def validate_action(action: NegotiationAction, current_contract: ContractDocument) -> bool:
    # Apply action to temporary contract copy
    temp_contract = current_contract.copy()
    temp_contract.apply_action(action)
    
    # Validate resulting contract
    validator = ContractValidator()
    result = validator.validate_contract(temp_contract.to_dict())
    
    return result.is_valid
```

#### 3. Tournament Validation Pipeline
```python
def tournament_validation_pipeline(contracts: List[ContractDocument]) -> ValidationReport:
    validator = ContractValidator()
    report = ValidationReport()
    
    for contract in contracts:
        result = validator.validate_contract(contract.to_dict())
        report.add_result(contract.id, result)
        
        # Flag contracts with issues for manual review
        if not result.is_legally_complete:
            report.flag_for_review(contract.id, result.legal_issues)
    
    return report
```

### Validation Dependencies
```python
# requirements.txt additions
jsonschema>=4.0.0
python-dateutil>=2.8.0
```

### Configuration
```python
# config/validation_config.py
VALIDATION_CONFIG = {
    "strict_mode": True,  # Fail on any validation error
    "require_legal_completeness": True,  # Require all legal elements
    "min_article_words": 10,  # Minimum words per article
    "max_articles": 20,  # Maximum number of articles
    "allowed_article_types": [
        "price_terms", "delivery_terms", "warranty_terms", 
        "payment_terms", "termination_clauses", "dispute_resolution",
        "governing_law", "liability_terms", "intellectual_property"
    ]
}
```

## LLM Interaction Protocol

### Overview

The system must handle interaction with LLMs in a **structured, controlled manner** to ensure reliable, parseable outputs. Modern LLMs work best when given clear tool/function definitions rather than free-form prompting.

### Design Philosophy

- **Structured Input/Output**: Contract state presented as JSON, actions returned as JSON
- **Tool-Based Interface**: LLMs select from predefined actions (similar to function calling)
- **Robust Parsing**: Multiple fallback strategies for handling model outputs
- **Model Agnostic**: Works with both API-based models (OpenAI) and local models (HuggingFace)

### Action Schema Definition

The system defines a formal JSON schema for all possible actions:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NegotiationAction",
  "type": "object",
  "properties": {
    "action_type": {
      "type": "string",
      "enum": ["EDIT_ARTICLE", "ADD_ARTICLE", "REMOVE_ARTICLE", "PASS"]
    },
    "article_name": {
      "type": "string",
      "description": "Name of the article to edit/add/remove"
    },
    "content": {
      "type": "string",
      "description": "New content for the article (for EDIT/ADD actions)"
    },
    "reasoning": {
      "type": "string",
      "description": "Explanation for why you're taking this action"
    }
  },
  "required": ["action_type"],
  "allOf": [
    {
      "if": {
        "properties": {"action_type": {"const": "EDIT_ARTICLE"}}
      },
      "then": {
        "required": ["article_name", "content", "reasoning"]
      }
    },
    {
      "if": {
        "properties": {"action_type": {"const": "ADD_ARTICLE"}}
      },
      "then": {
        "required": ["article_name", "content", "reasoning"]
      }
    },
    {
      "if": {
        "properties": {"action_type": {"const": "REMOVE_ARTICLE"}}
      },
      "then": {
        "required": ["article_name", "reasoning"]
      }
    },
    {
      "if": {
        "properties": {"action_type": {"const": "PASS"}}
      },
      "then": {
        "required": ["reasoning"]
      }
    }
  ]
}
```

### Prompt Structure

#### System Prompt Template

```python
SYSTEM_PROMPT = """You are an AI contract negotiation agent. Your role is to negotiate contracts on behalf of your client by selecting strategic actions.

You will receive:
1. The current contract state (as JSON)
2. Your role (BUYER or SELLER)
3. Your remaining edit budget
4. Number of remaining turns

You must respond with a valid JSON action following the exact schema provided.

Available actions:
- EDIT_ARTICLE: Modify an existing contract article (cost: 1)
- ADD_ARTICLE: Add a new contract article (cost: 2)
- REMOVE_ARTICLE: Remove an existing article (cost: 2)
- PASS: Make no changes this turn (cost: 0)

Strategic considerations:
- You have limited edit budget - use it wisely
- The negotiation runs for a fixed number of turns
- Free-form writing is allowed within articles, but you can only modify one article per action
- Changes are tracked and visible to the other party
- Use PASS when you're satisfied with the current contract

Always provide reasoning for your actions to demonstrate strategic thinking.
"""
```

#### Turn Prompt Template

```python
def build_negotiation_prompt(
    contract: ContractDocument,
    role: str,
    remaining_budget: int,
    remaining_turns: int,
    history: List[TurnRecord]
) -> str:
    """Build the full prompt for a negotiation turn."""

    # Highlight recent changes
    recent_changes = format_recent_changes(history, last_n=2)

    prompt = f"""
CURRENT CONTRACT STATE:
{json.dumps(contract.to_dict(), indent=2)}

YOUR SITUATION:
- Role: {role}
- Remaining edit budget: {remaining_budget}
- Remaining turns: {remaining_turns}

RECENT CHANGES BY OTHER PARTY:
{recent_changes}

AVAILABLE ARTICLES YOU CAN MODIFY:
{', '.join(contract.articles.keys())}

RESPOND WITH VALID JSON ACTION:
{{
  "action_type": "EDIT_ARTICLE" | "ADD_ARTICLE" | "REMOVE_ARTICLE" | "PASS",
  "article_name": "price_terms",  // required for EDIT/ADD/REMOVE
  "content": "The revised article text...",  // required for EDIT/ADD
  "reasoning": "Strategic explanation for this action"  // always required
}}

Your JSON response:
"""
    return prompt
```

### Implementation Strategies

#### Strategy 1: Native Function Calling (for supported models)

For models that support OpenAI-style function calling (GPT-4, Claude, Gemini):

```python
# infrastructure/llm/function_calling_agent.py
class FunctionCallingLawyerAgent(LawyerAgent):
    """Uses native function calling APIs."""

    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "edit_article",
                "description": "Modify an existing contract article",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "article_name": {
                            "type": "string",
                            "description": "Name of article to edit"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New text for the article"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why you're making this change"
                        }
                    },
                    "required": ["article_name", "new_content", "reasoning"]
                }
            }
        },
        # ... other tools (add_article, remove_article, pass)
    ]

    def select_action(self, contract, remaining_budget, remaining_turns):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_situation(contract, remaining_budget, remaining_turns)}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.TOOL_DEFINITIONS,
            tool_choice="required"  # Must call a tool
        )

        tool_call = response.choices[0].message.tool_calls[0]
        return self._convert_tool_call_to_action(tool_call)
```

#### Strategy 2: JSON Mode with Schema (for HuggingFace models)

For open-source models via HuggingFace:

```python
# infrastructure/llm/json_mode_agent.py
import json
import jsonschema
from transformers import AutoModelForCausalLM, AutoTokenizer

class JSONModeLawyerAgent(LawyerAgent):
    """Uses JSON mode or constrained generation."""

    def __init__(self, model_name: str, use_constrained_generation: bool = True):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.use_constrained_generation = use_constrained_generation

        if use_constrained_generation:
            # Use outlines or guidance for guaranteed valid JSON
            from outlines import models, generate
            self.constrained_model = models.Transformers(model_name)
            self.generator = generate.json(
                self.constrained_model,
                schema=NegotiationAction  # Pydantic model
            )

    def select_action(self, contract, remaining_budget, remaining_turns):
        prompt = self.prompt_builder.build_negotiation_prompt(
            contract, self.role, remaining_budget, remaining_turns
        )

        if self.use_constrained_generation:
            # Guaranteed valid JSON
            action_dict = self.generator(prompt)
        else:
            # Best-effort JSON parsing
            response = self._generate_text(prompt)
            action_dict = self._extract_json(response)

        # Validate against schema
        jsonschema.validate(action_dict, ACTION_SCHEMA)

        return self._parse_action(action_dict)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from model response (handles various formats)."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try finding JSON block
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not extract valid JSON from response: {text}")
```

#### Strategy 3: Robust Prompt Engineering (fallback)

For models without special JSON support:

```python
# infrastructure/llm/prompt_based_agent.py
class PromptBasedLawyerAgent(LawyerAgent):
    """Uses careful prompt engineering and robust parsing."""

    def select_action(self, contract, remaining_budget, remaining_turns):
        prompt = f"""You are negotiating a contract. Choose ONE action.

CONTRACT:
{self._format_contract_for_display(contract)}

YOUR ROLE: {self.role}
BUDGET: {remaining_budget} edits remaining
TURNS: {remaining_turns} turns remaining

RESPOND IN THIS EXACT FORMAT:

ACTION: [EDIT_ARTICLE | ADD_ARTICLE | REMOVE_ARTICLE | PASS]
ARTICLE: [article name, or N/A for PASS]
CONTENT: [new article text, or N/A for REMOVE/PASS]
REASONING: [your strategic reasoning]

Example valid response:
ACTION: EDIT_ARTICLE
ARTICLE: price_terms
CONTENT: The buyer shall pay $450 per unit, payable within 30 days of delivery.
REASONING: Negotiating price down from $500 to improve buyer position while maintaining reasonable payment terms.

Your response:
"""

        response = self._generate_text(prompt)
        return self._parse_structured_response(response)

    def _parse_structured_response(self, response: str) -> NegotiationAction:
        """Parse structured text response into action object."""
        lines = response.strip().split('\n')
        fields = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                fields[key.strip().lower()] = value.strip()

        action_type = fields.get('action', '').upper()

        if action_type == 'EDIT_ARTICLE':
            return EditArticleAction(
                article_name=fields['article'],
                new_content=fields['content'],
                reasoning=fields['reasoning']
            )
        elif action_type == 'PASS':
            return PassAction(reasoning=fields['reasoning'])
        # ... etc

        raise ValueError(f"Could not parse action from response: {response}")
```

### Error Handling

```python
class LLMAgentErrorHandler:
    """Handles common LLM response errors."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def execute_with_retry(
        self,
        action_fn: Callable,
        validation_fn: Callable
    ) -> NegotiationAction:
        """Try to get valid action, with retries."""

        for attempt in range(self.max_retries):
            try:
                result = action_fn()
                validation_fn(result)
                return result

            except jsonschema.ValidationError as e:
                if attempt < self.max_retries - 1:
                    # Add error feedback to next prompt
                    error_msg = f"Previous response was invalid: {e.message}. Please try again."
                    # Retry with error feedback
                    continue
                else:
                    # Final attempt failed - use fallback
                    return self._get_fallback_action()

            except Exception as e:
                logging.error(f"LLM action generation failed: {e}")
                if attempt < self.max_retries - 1:
                    continue
                else:
                    return self._get_fallback_action()

        return self._get_fallback_action()

    def _get_fallback_action(self) -> NegotiationAction:
        """Safe fallback when LLM fails."""
        return PassAction(reasoning="Error in action generation, passing turn")
```

### Model-Specific Configurations

```python
# config/model_config.py
MODEL_CONFIGS = {
    "gpt-4": {
        "strategy": "function_calling",
        "supports_json_mode": True,
        "max_tokens": 2000,
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "strategy": "json_mode",
        "supports_json_mode": True,  # With right prompting
        "use_constrained_generation": False,  # Too slow
        "max_tokens": 2000,
    },
    "meta-llama/Llama-3-8b-chat-hf": {
        "strategy": "json_mode",
        "supports_json_mode": True,
        "use_constrained_generation": False,
        "max_tokens": 2000,
    },
    "AdaptLLM/law-chat": {
        "strategy": "prompt_based",  # Older model, needs structured prompts
        "supports_json_mode": False,
        "max_tokens": 1500,
    }
}
```

### Testing Strategy

```python
# tests/test_llm_interaction.py
def test_action_generation_with_mock():
    """Test action generation without real LLM."""

    mock_response = {
        "action_type": "EDIT_ARTICLE",
        "article_name": "price_terms",
        "content": "The buyer shall pay $450 per unit.",
        "reasoning": "Negotiating better price"
    }

    agent = MockLawyerAgent(mock_response)
    action = agent.select_action(contract, budget=5, turns=3)

    assert isinstance(action, EditArticleAction)
    assert action.article_name == "price_terms"

def test_json_schema_validation():
    """Ensure all action types validate correctly."""

    valid_actions = [
        {"action_type": "EDIT_ARTICLE", "article_name": "price_terms", "content": "...", "reasoning": "..."},
        {"action_type": "PASS", "reasoning": "Satisfied with contract"},
    ]

    for action_dict in valid_actions:
        jsonschema.validate(action_dict, ACTION_SCHEMA)  # Should not raise

def test_error_recovery():
    """Test that invalid responses are handled gracefully."""

    handler = LLMAgentErrorHandler()

    # Mock LLM that returns invalid JSON first, then valid
    attempts = [
        "This is not JSON",  # Attempt 1: invalid
        '{"action_type": "PASS", "reasoning": "OK"}',  # Attempt 2: valid
    ]

    action = handler.execute_with_retry(...)
    assert isinstance(action, PassAction)
```

### Key Dependencies

```python
# requirements.txt additions
jsonschema>=4.0.0           # Schema validation
outlines>=0.0.20            # Constrained generation (optional)
guidance>=0.1.0             # Structured generation (optional)
pydantic>=2.0.0             # Type validation
```

## Evaluation Framework

### Multi-Dimensional Scoring

#### Contract Quality Metrics (0-100 points each)
- **Legal Completeness**: Presence of required legal elements
- **Commercial Reasonableness**: Market-standard terms and conditions
- **Risk Balance**: Fair allocation of risks between parties
- **Enforceability**: Likelihood of successful legal enforcement
- **Clarity**: Unambiguous language and defined terms

#### Negotiation Process Metrics
- **Strategic Sophistication**: Quality of negotiation moves and timing
- **Efficiency**: Achieving goals with minimal edit budget
- **Responsiveness**: Appropriate reactions to counterparty actions
- **Budget Management**: How agents allocate limited resources across turns
- **Convergence Behavior**: Turn at which agents begin to PASS (implicit satisfaction signal)
- **Edit Patterns**: Size and frequency of modifications over time

#### Judge Panel Configuration
- Primary judges: 6 AI models not participating in current negotiation
- Secondary validation: Specialized legal AI models
- Ground truth: Human legal expert validation (sample subset)

---

# APPENDIX: Future Enhancements & Detailed Design

**Note**: The sections below represent the "ideal" architecture for a production-quality system. These are NOT required for the initial implementation but serve as:
1. Reference for what "good" looks like
2. Guidance for future refactoring
3. Solutions to problems you'll likely encounter

**Use this appendix when:**
- The simple implementation hits limitations
- You need to scale beyond 8 models
- Converting this into a benchmark/library for others
- Publishing results that need reproducibility guarantees

---

## Appendix A: Full Layered Architecture (Optional)

**When to implement**: When the simple flat structure becomes hard to maintain or test.

### Detailed Architecture Overview

The system CAN follow a **three-layer architecture** with clear separation of concerns (if needed):

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  ┌──────────────────────┐        ┌─────────────────────┐        │
│  │ Tournament           │───────►│ Results             │        │
│  │ Orchestrator         │        │ Analyzer            │        │
│  │                      │        │                     │        │
│  │ - Match scheduling   │        │ - Statistical       │        │
│  │ - Round-robin logic  │        │   analysis          │        │
│  │ - Role assignment    │        │ - Report generation │        │
│  └──────────────────────┘        └─────────────────────┘        │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│                       DOMAIN LAYER                               │
│  ┌────────────────────────────────────────────────┐              │
│  │ NegotiationSession                             │              │
│  │ ┌────────────────┐  ┌─────────────────────┐   │              │
│  │ │ TurnManager    │  │ ChangeTracker       │   │              │
│  │ │                │  │                     │   │              │
│  │ │ - Execute turn │  │ - Record diffs      │   │              │
│  │ │ - Track budget │  │ - History tracking  │   │              │
│  │ └────────────────┘  └─────────────────────┘   │              │
│  └────────────────────────────────────────────────┘              │
│                                                                   │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ ContractDocument     │   │ ContractValidator       │         │
│  │                      │   │                         │         │
│  │ - Articles           │   │ - Schema validation     │         │
│  │ - Metadata           │   │ - Business rules        │         │
│  │ - to_dict()          │   │ - Legal completeness    │         │
│  └──────────────────────┘   └─────────────────────────┘         │
│                                                                   │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ LawyerAgent          │   │ JudgePanel              │         │
│  │ (Abstract Interface) │   │ (Abstract Interface)    │         │
│  │                      │   │                         │         │
│  │ + select_action()    │   │ + evaluate_contract()   │         │
│  └──────────────────────┘   └─────────────────────────┘         │
└──────────────┬──────────────────────┬───────────────────────────┘
               │                      │
┌──────────────▼──────────────────────▼───────────────────────────┐
│                   INFRASTRUCTURE LAYER                           │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ LLMLawyerAgent       │   │ LLMJudgePanel           │         │
│  │ (Concrete)           │   │ (Concrete)              │         │
│  │                      │   │                         │         │
│  │ - Build prompts      │   │ - Scoring prompts       │         │
│  │ - Parse JSON actions │   │ - Multi-model ensemble  │         │
│  │ - Handle errors      │   │                         │         │
│  └──────────┬───────────┘   └─────────────────────────┘         │
│             │                                                    │
│  ┌──────────▼───────────┐   ┌─────────────────────────┐         │
│  │ ModelManager         │   │ DataStorage             │         │
│  │                      │   │                         │         │
│  │ - GPU allocation     │   │ - ContractRepository    │         │
│  │ - Model loading      │   │ - ResultsRepository     │         │
│  │ - Memory management  │   │ - JSON/CSV persistence  │         │
│  └──────────────────────┘   └─────────────────────────┘         │
│                                                                   │
│  ┌──────────────────────┐   ┌─────────────────────────┐         │
│  │ NegotiationLogger    │   │ ConfigurationManager    │         │
│  │                      │   │                         │         │
│  │ - Structured logs    │   │ - Experiment configs    │         │
│  │ - Event tracking     │   │ - Model parameters      │         │
│  └──────────────────────┘   └─────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Full Module Structure (If Needed)

```
sandlex/
├── domain/                      # Core business logic (no external dependencies)
│   ├── contracts/
│   │   ├── contract_document.py      # ContractDocument, Article classes
│   │   ├── contract_validator.py     # Validation logic
│   │   └── actions.py                # NegotiationAction types
│   ├── negotiation/
│   │   ├── negotiation_session.py    # Orchestrates one negotiation
│   │   ├── turn_manager.py           # Turn execution, budget tracking
│   │   └── change_tracker.py         # Records diffs between turns
│   └── agents/
│       ├── lawyer_agent.py           # Abstract LawyerAgent interface
│       └── judge_panel.py            # Abstract JudgePanel interface
│
├── application/                 # Use cases and orchestration
│   ├── tournament/
│   │   ├── tournament_orchestrator.py
│   │   └── match_scheduler.py
│   └── analysis/
│       └── results_analyzer.py
│
├── infrastructure/              # Technical implementation details
│   ├── llm/
│   │   ├── model_manager.py          # GPU allocation, model loading
│   │   ├── llm_lawyer_agent.py       # Concrete LawyerAgent using LLMs
│   │   ├── llm_judge_panel.py        # Concrete JudgePanel using LLMs
│   │   └── prompt_builder.py         # Prompt engineering utilities
│   ├── storage/
│   │   ├── contract_repository.py    # Save/load contracts
│   │   └── results_repository.py     # Save/load results
│   └── logging/
│       └── negotiation_logger.py     # Structured logging
│
└── config/
    ├── experiment_config.py     # Experiment parameters
    └── model_config.py          # Model-specific settings
```

(See full class implementation details below...)

## Appendix B: Advanced LLM Interaction Strategies (Optional)

**When to implement**: When simple JSON parsing fails frequently or you need to support multiple model types.

### Problem Statement
The minimal implementation uses basic prompt + JSON parsing. This may fail when:
- Models don't follow JSON format reliably
- You want to use API-based models (GPT-4, Claude)
- You need guaranteed valid outputs
- Different models need different prompting strategies

### Solution: Multiple Parsing Strategies

(Full LLM interaction protocol details below...)

## Appendix C: Complete Validation Framework (Optional)

**When to implement**: When you need rigorous contract quality checks or publishing results.

(JSON schema validation details below...)

## Appendix D: Multi-Dimensional Evaluation (Optional)

**When to implement**: When simple win/loss isn't enough for analysis.

(Full evaluation framework below...)

---

## Original Implementation Phases (12-week plan)

**Note**: This timeline is for the FULL system with all bells and whistles. The minimal version (above) takes 4-6 weeks.

### Phase 1: Core Infrastructure (Weeks 1-4)
- [ ] Implement ContractDocument and ContractEngine classes
- [ ] Create turn management system with budget tracking
- [ ] Develop contract validation framework
- [ ] Build change tracking and diff visualization

### Phase 2: Agent Integration (Weeks 5-6)
- [ ] Redesign LawyerAgent interface for structured actions
- [ ] Implement action validation and execution
- [ ] Create agent prompt templates for new system
- [ ] Test individual agent functionality

### Phase 3: Tournament System (Weeks 7-8)
- [ ] Build tournament orchestration framework
- [ ] Implement judge panel management
- [ ] Create comprehensive logging system
- [ ] Develop results analysis pipeline

### Phase 4: Evaluation Enhancement (Weeks 9-10)
- [ ] Implement multi-dimensional scoring system
- [ ] Create statistical analysis framework
- [ ] Build reporting and visualization tools
- [ ] Validate against human expert assessments

### Phase 5: Testing and Refinement (Weeks 11-12)
- [ ] Comprehensive system testing
- [ ] Performance optimization
- [ ] Documentation completion
- [ ] Baseline experiment execution

## Status Tracking

### Overall Project Status
**Current Phase**: Planning Complete
**Overall Progress**: 0% Implementation
**Target Completion**: 12 weeks from start
**Risk Level**: Medium

### Phase Status

#### Phase 1: Core Infrastructure
- **Status**: Not Started
- **Progress**: 0%
- **Blockers**: None identified
- **Next Actions**: Begin ContractDocument class implementation
- **Estimated Completion**: Week 4

#### Phase 2: Agent Integration  
- **Status**: Not Started
- **Progress**: 0%
- **Blockers**: Dependent on Phase 1 completion
- **Next Actions**: Pending Phase 1
- **Estimated Completion**: Week 6

#### Phase 3: Tournament System
- **Status**: Not Started
- **Progress**: 0% 
- **Blockers**: Dependent on Phase 2 completion
- **Next Actions**: Pending Phase 2
- **Estimated Completion**: Week 8

#### Phase 4: Evaluation Enhancement
- **Status**: Not Started
- **Progress**: 0%
- **Blockers**: Dependent on Phase 3 completion
- **Next Actions**: Pending Phase 3
- **Estimated Completion**: Week 10

#### Phase 5: Testing and Refinement
- **Status**: Not Started
- **Progress**: 0%
- **Blockers**: Dependent on Phase 4 completion
- **Next Actions**: Pending Phase 4
- **Estimated Completion**: Week 12

### Technical Risks and Mitigation

#### High Risk Items
1. **GPU Resource Constraints**: 8 models may exceed available VRAM
   - *Mitigation*: Implement model rotation and offloading strategies
   
2. **Judge Response Quality**: Current system shows corrupted judge outputs
   - *Mitigation*: Redesign prompts with strict output formatting

3. **Contract Validation Complexity**: Legal validation may be computationally expensive
   - *Mitigation*: Implement tiered validation with basic checks first

#### Medium Risk Items
1. **Agent Adaptation Time**: Models may need multiple examples to understand new format
   - *Mitigation*: Create comprehensive few-shot prompting examples

2. **Statistical Power**: May need larger sample sizes for significant results
   - *Mitigation*: Implement power analysis and adaptive sample sizing

### Change Log

#### Version 1.0 (Initial Plan)
- **Date**: [Current Date]
- **Changes**: Initial comprehensive redesign plan created
- **Author**: Strategic Planning Agent
- **Next Review**: After Phase 1 completion

---

## Notes for Implementation Team

### Key Dependencies
- PyTorch and Transformers library compatibility
- Available GPU memory for concurrent model loading
- **JSON schema validation**: `jsonschema>=4.0.0`, `python-dateutil>=2.8.0`
- Statistical analysis framework (scipy/statsmodels)
- Data manipulation: `pandas>=1.3.0`
- Configuration management: `pydantic>=1.8.0` (for settings validation)

### Success Criteria
- 95%+ contract validation accuracy
- <5% judge response format errors
- Reproducible results with statistical significance
- Human expert correlation >0.7 for contract quality scores

### Future Enhancements
- Multi-party negotiations (buyer, seller, financier)
- Dynamic contract templates based on complexity
- Real-time strategy analysis and recommendation
- Integration with real legal document databases