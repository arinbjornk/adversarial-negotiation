"""
Negotiation session management and turn-based protocol.

This module handles the turn-by-turn execution of contract negotiations
between buyer and seller agents.
"""

import json
from datetime import datetime
from typing import List, Dict
from contracts import ContractDocument, get_action_cost
from agents import LLMAgent


class TurnManager:
    """
    Manages turn order, budgets, and completion conditions.

    The TurnManager enforces the fixed 6-turn protocol and tracks
    each agent's remaining edit budget.
    """

    def __init__(self, total_turns: int = 6, budget_per_agent: int = 10):
        """
        Initialize turn manager.

        Args:
            total_turns: Total number of turns (default 6)
            budget_per_agent: Edit budget for each agent (default 10)
        """
        self.total_turns = total_turns
        self.budget_per_agent = budget_per_agent
        self.current_turn = 0
        self.budgets: Dict[str, int] = {}

    def initialize_agent_budget(self, agent_id: str) -> None:
        """Initialize budget for an agent."""
        self.budgets[agent_id] = self.budget_per_agent

    def get_budget(self, agent_id: str) -> int:
        """Get remaining budget for an agent."""
        return self.budgets.get(agent_id, 0)

    def deduct_budget(self, agent_id: str, cost: int) -> bool:
        """
        Deduct cost from agent's budget.

        Returns:
            True if budget was sufficient, False otherwise
        """
        current_budget = self.budgets.get(agent_id, 0)
        if current_budget >= cost:
            self.budgets[agent_id] = current_budget - cost
            return True
        return False

    def is_complete(self) -> bool:
        """Check if negotiation is complete."""
        return self.current_turn >= self.total_turns

    def increment_turn(self) -> None:
        """Advance to next turn."""
        self.current_turn += 1

    def remaining_turns(self) -> int:
        """Get number of turns remaining."""
        return self.total_turns - self.current_turn


class NegotiationSession:
    """
    Orchestrates a single negotiation between buyer and seller.

    The session runs a fixed number of alternating turns and tracks
    all actions taken by both agents.
    """

    def __init__(
        self,
        buyer: LLMAgent,
        seller: LLMAgent,
        initial_contract: ContractDocument,
        turn_manager: TurnManager
    ):
        """
        Initialize a negotiation session.

        Args:
            buyer: The buyer agent
            seller: The seller agent
            initial_contract: Starting contract
            turn_manager: Turn management system
        """
        self.buyer = buyer
        self.seller = seller
        self.contract = initial_contract
        self.turn_manager = turn_manager

        # Alternating turn order: buyer, seller, buyer, seller, buyer, seller
        self.turn_order = [buyer, seller] * (turn_manager.total_turns // 2)

        # Initialize budgets
        turn_manager.initialize_agent_budget(buyer.agent_id)
        turn_manager.initialize_agent_budget(seller.agent_id)

        # Track all actions
        self.action_log: List[dict] = []

    def run(self) -> Dict:
        """
        Execute the full negotiation.

        Returns:
            Dictionary with final contract, action log, and metadata
        """
        print(f"\n{'='*60}")
        print(f"NEGOTIATION: {self.buyer.agent_id} (buyer) vs {self.seller.agent_id} (seller)")
        print(f"{'='*60}\n")

        turn_number = 0

        while not self.turn_manager.is_complete():
            current_agent = self.turn_order[turn_number]

            print(f"Turn {turn_number + 1}/{self.turn_manager.total_turns} - {current_agent.agent_id} ({current_agent.role})")
            print(f"  Budget: {self.turn_manager.get_budget(current_agent.agent_id)} points remaining")

            # Get action from agent
            action = current_agent.select_action(
                contract=self.contract,
                remaining_budget=self.turn_manager.get_budget(current_agent.agent_id),
                remaining_turns=self.turn_manager.remaining_turns()
            )

            # Calculate cost
            cost = get_action_cost(action)
            print(f"  Action: {action.get('action_type')} (cost: {cost})")

            # Check budget
            if not self.turn_manager.deduct_budget(current_agent.agent_id, cost):
                print(f"  WARNING: Insufficient budget! Forcing PASS.")
                action = {"action_type": "PASS", "reasoning": "Insufficient budget"}
                cost = 0

            # Apply action to contract
            self.contract.apply_action(action, current_agent.agent_id)

            # Log the action
            action_record = {
                "turn": turn_number + 1,
                "agent_id": current_agent.agent_id,
                "role": current_agent.role,
                "action": action,
                "cost": cost,
                "remaining_budget": self.turn_manager.get_budget(current_agent.agent_id),
                "timestamp": datetime.now().isoformat()
            }
            self.action_log.append(action_record)

            if action.get("reasoning"):
                print(f"  Reasoning: {action.get('reasoning')[:100]}...")

            print()

            # Move to next turn
            self.turn_manager.increment_turn()
            turn_number += 1

        print(f"{'='*60}")
        print(f"NEGOTIATION COMPLETE")
        print(f"{'='*60}\n")

        return {
            "buyer_id": self.buyer.agent_id,
            "seller_id": self.seller.agent_id,
            "final_contract": self.contract.to_dict(),
            "action_log": self.action_log,
            "buyer_final_budget": self.turn_manager.get_budget(self.buyer.agent_id),
            "seller_final_budget": self.turn_manager.get_budget(self.seller.agent_id),
            "total_turns": self.turn_manager.total_turns,
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, filepath: str) -> None:
        """Save negotiation results to JSON file."""
        results = self.run() if not self.action_log else {
            "buyer_id": self.buyer.agent_id,
            "seller_id": self.seller.agent_id,
            "final_contract": self.contract.to_dict(),
            "action_log": self.action_log,
            "buyer_final_budget": self.turn_manager.get_budget(self.buyer.agent_id),
            "seller_final_budget": self.turn_manager.get_budget(self.seller.agent_id),
            "total_turns": self.turn_manager.total_turns,
            "timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
