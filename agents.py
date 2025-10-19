"""
LLM-based negotiation agents for contract competition.

This module provides the LLMAgent class which uses transformer models
to make strategic negotiation decisions.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from contracts import ContractDocument


class LLMAgent:
    """
    An LLM-based agent that negotiates contracts.

    The agent uses a transformer model to analyze contracts and select
    actions (EDIT, ADD, REMOVE, PASS) based on its role (buyer/seller).
    """

    # Class-level GPU management
    _gpu_index = 0
    _num_gpus = torch.cuda.device_count()

    def __init__(self, model_name: str, role: str, agent_id: Optional[str] = None):
        """
        Initialize an LLM agent.

        Args:
            model_name: HuggingFace model identifier
            role: Either 'buyer' or 'seller'
            agent_id: Optional ID for the agent (defaults to model_name)
        """
        self.model_name = model_name
        self.role = role
        self.agent_id = agent_id or model_name

        # Assign GPU (cycle through available GPUs)
        self.device = torch.device(
            f"cuda:{LLMAgent._gpu_index}" if torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            LLMAgent._gpu_index = (LLMAgent._gpu_index + 1) % max(1, LLMAgent._num_gpus)

        # Load model and tokenizer
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None  # We handle device placement manually
        ).to(self.device)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def select_action(
        self,
        contract: ContractDocument,
        remaining_budget: int,
        remaining_turns: int
    ) -> dict:
        """
        Select a negotiation action based on current contract state.

        Args:
            contract: Current contract document
            remaining_budget: Number of edit points remaining
            remaining_turns: Number of turns left in negotiation

        Returns:
            Dictionary representing the action to take
        """
        prompt = self._build_prompt(contract, remaining_budget, remaining_turns)

        try:
            response_text = self._generate_response(prompt)
            action = self._parse_json_response(response_text)
            return action
        except Exception as e:
            print(f"Error generating action for {self.agent_id}: {e}")
            # Fallback to PASS on any error
            return {
                "action_type": "PASS",
                "reasoning": f"Error in action generation: {str(e)}"
            }

    def _build_prompt(
        self,
        contract: ContractDocument,
        remaining_budget: int,
        remaining_turns: int
    ) -> str:
        """Build the prompt for the LLM."""

        # Get contract as dict
        contract_dict = contract.to_dict()

        # Show recent changes (last 2 actions)
        recent_changes = ""
        if len(contract.history) > 0:
            recent = contract.history[-2:] if len(contract.history) >= 2 else contract.history
            recent_changes = "\n".join([
                f"- {h['agent']} performed {h['action']['action_type']}"
                for h in recent
            ])
        else:
            recent_changes = "No changes yet (initial contract)"

        prompt = f"""You are a contract negotiation agent representing the {self.role.upper()}.

CURRENT CONTRACT:
{json.dumps(contract_dict, indent=2)}

YOUR SITUATION:
- Role: {self.role.upper()}
- Remaining edit budget: {remaining_budget} points
- Remaining turns: {remaining_turns}

RECENT CHANGES:
{recent_changes}

AVAILABLE ARTICLES YOU CAN MODIFY:
{', '.join(contract.list_articles())}

YOUR TASK:
As the {self.role}, negotiate to improve this contract in your favor. You can:
- EDIT_ARTICLE (cost: 1) - Modify an existing article
- ADD_ARTICLE (cost: 2) - Add a new article to the contract
- REMOVE_ARTICLE (cost: 2) - Remove an article
- PASS (cost: 0) - Accept the current contract

Respond with ONLY a valid JSON object in this exact format:
{{
  "action_type": "EDIT_ARTICLE",
  "article_name": "price_terms",
  "content": "The new article text goes here...",
  "reasoning": "Brief explanation of why you're making this change"
}}

For PASS action:
{{
  "action_type": "PASS",
  "reasoning": "Why you're accepting the current contract"
}}

Your JSON response:
"""
        return prompt

    def _generate_response(self, prompt: str) -> str:
        """Generate response from the LLM."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=500,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,  # Deterministic for reproducibility
            )

        # Decode only the generated tokens (not the prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def _parse_json_response(self, text: str) -> dict:
        """
        Parse JSON from LLM response.

        Handles various formats and falls back to PASS on parse errors.
        """
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try finding JSON block in text
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                # Validate it has action_type
                if "action_type" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        # If we get here, parsing failed
        raise ValueError(f"Could not parse JSON from response: {text[:200]}...")
