# Contract Competition System - Redesign Plan

## Project Overview

**Goal**: Transform the existing AI contract competition system into a scientifically rigorous experimental platform for studying AI negotiation capabilities.

**Current System Problems**:
- Unstructured contract editing leading to inconsistent results
- Poor judgment quality with corrupted responses
- No change tracking between negotiation turns
- Binary evaluation inadequate for contract complexity
- Lack of experimental controls and reproducibility

## Requirements

### Functional Requirements

#### Core Negotiation System
- **FR-1**: Structured contract representation using JSON format with standard articles
- **FR-2**: Atomic contract operations (edit, add, remove articles)
- **FR-3**: Edit budget system to prevent infinite negotiations
- **FR-4**: Change tracking with diff visualization between turns
- **FR-5**: Contract validation for legal completeness
- **FR-6**: Multi-agent turn-based negotiation protocol

#### Evaluation System
- **FR-7**: Multi-dimensional contract scoring (quality, fairness, enforceability)
- **FR-8**: Judge blinding to prevent model bias
- **FR-9**: Inter-rater reliability measurement
- **FR-10**: Statistical confidence reporting

#### Tournament Management
- **FR-11**: Round-robin tournament with role randomization
- **FR-12**: Standardized baseline contracts for fair comparison
- **FR-13**: Comprehensive logging of all negotiations
- **FR-14**: Results export with statistical analysis

### Non-Functional Requirements
- **NFR-1**: Support 8+ concurrent AI models on available GPU resources
- **NFR-2**: Reproducible experiments with seed control
- **NFR-3**: Extensible architecture for new contract types
- **NFR-4**: Real-time progress monitoring and intervention capability

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Tournament     │    │  Negotiation    │    │  Contract       │
│  Orchestrator   │◄──►│  Engine         │◄──►│  Validator      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Results        │    │  Change         │    │  Judge          │
│  Analyzer       │    │  Tracker        │    │  Panel          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Storage   │    │  Logging        │    │  AI Model       │
│  Layer          │    │  System         │    │  Manager        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Class Structure

```python
# Core Classes
class ContractEngine:
    - contract_state: ContractDocument
    - edit_budget: Dict[agent_id, int]
    - change_history: List[ContractChange]
    
class ContractDocument:
    - articles: Dict[str, Article]
    - metadata: ContractMetadata
    - validation_status: ValidationResult
    
class NegotiationSession:
    - agents: List[LawyerAgent]
    - contract: ContractDocument
    - turn_manager: TurnManager
    - results: NegotiationResult

class LawyerAgent:
    - model: LanguageModel
    - role: AgentRole (BUYER/SELLER)
    - action_history: List[NegotiationAction]
```

## Data Flows

### 1. Tournament Initialization Flow
```
Tournament Start → Load Base Contracts → Initialize Agent Pairs → 
Create Judge Panels → Begin Negotiation Sessions
```

### 2. Negotiation Turn Flow
```
Agent Receives Contract → Analyzes Changes → Selects Action → 
Validates Action → Updates Contract → Records Change → 
Checks Budget/Convergence → Next Turn or End
```

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
    def execute_turn(self, agent: LawyerAgent, contract: ContractDocument) -> TurnResult:
        # 1. Present current contract state with changes highlighted
        contract_view = self.prepare_contract_view(contract, agent.role)
        
        # 2. Get agent action with budget constraints
        action = agent.select_action(contract_view, self.get_budget(agent))
        
        # 3. Validate and apply action
        if self.validate_action(action, agent):
            updated_contract = self.apply_action(contract, action)
            self.deduct_budget(agent, action.cost)
            self.record_change(action, agent)
            return TurnResult.SUCCESS(updated_contract)
        else:
            return TurnResult.INVALID_ACTION(action.validation_error)
```

### Action Types and Costs

| Action Type | Edit Cost | Description |
|-------------|-----------|-------------|
| EDIT_ARTICLE | 1 | Modify existing article content |
| ADD_ARTICLE | 2 | Add new contract article |
| REMOVE_ARTICLE | 2 | Remove existing article |
| ACCEPT_CONTRACT | 0 | Accept current contract terms |
| REQUEST_CLARIFICATION | 1 | Ask for specific clarification |

### Convergence Conditions
1. Both agents execute ACCEPT_CONTRACT
2. Edit budgets exhausted for both agents
3. Maximum turn limit reached (safety mechanism)
4. Contract reaches minimum viability threshold

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

## Evaluation Framework

### Multi-Dimensional Scoring

#### Contract Quality Metrics (0-100 points each)
- **Legal Completeness**: Presence of required legal elements
- **Commercial Reasonableness**: Market-standard terms and conditions
- **Risk Balance**: Fair allocation of risks between parties
- **Enforceability**: Likelihood of successful legal enforcement
- **Clarity**: Unambiguous language and defined terms

#### Negotiation Process Metrics
- **Strategic Sophistication**: Quality of negotiation moves
- **Efficiency**: Achieving goals with minimal edit budget
- **Responsiveness**: Appropriate reactions to counterparty actions

#### Judge Panel Configuration
- Primary judges: 6 AI models not participating in current negotiation
- Secondary validation: Specialized legal AI models
- Ground truth: Human legal expert validation (sample subset)

## Implementation Phases

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