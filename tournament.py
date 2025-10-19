"""
Tournament management for running round-robin negotiations.

This module handles pairing agents, running matches, and collecting results.
"""

import random
import pandas as pd
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from contracts import ContractDocument
from agents import LLMAgent
from negotiation import TurnManager, NegotiationSession


def create_round_robin_matches(agents: List[LLMAgent]) -> List[tuple]:
    """
    Create all possible pairings for round-robin tournament.

    Args:
        agents: List of LLMAgent instances

    Returns:
        List of (agent_a, agent_b) tuples
    """
    matches = []
    for i, agent_a in enumerate(agents):
        for agent_b in agents[i+1:]:
            matches.append((agent_a, agent_b))
    return matches


def run_match(
    agent_a: LLMAgent,
    agent_b: LLMAgent,
    baseline_contract: ContractDocument,
    turns: int = 6,
    budget: int = 10,
    save_dir: str = "results"
) -> Dict:
    """
    Run a single negotiation match between two agents.

    Args:
        agent_a: First agent
        agent_b: Second agent
        baseline_contract: Starting contract
        turns: Number of turns (default 6)
        budget: Budget per agent (default 10)
        save_dir: Directory to save results

    Returns:
        Dictionary with match results
    """
    # Randomly assign buyer/seller roles
    if random.random() < 0.5:
        buyer, seller = agent_a, agent_b
    else:
        buyer, seller = agent_b, agent_a

    print(f"\nMatch: {agent_a.agent_id} vs {agent_b.agent_id}")
    print(f"Roles: {buyer.agent_id} = BUYER, {seller.agent_id} = SELLER")

    # Create fresh copies for this negotiation
    buyer.role = "buyer"
    seller.role = "seller"

    # Load fresh contract for this match
    contract_copy = ContractDocument.from_dict(baseline_contract.to_dict())

    # Create turn manager
    turn_manager = TurnManager(total_turns=turns, budget_per_agent=budget)

    # Create and run negotiation
    session = NegotiationSession(buyer, seller, contract_copy, turn_manager)
    results = session.run()

    # Save negotiation log
    save_path = Path(save_dir)
    negotiation_file = save_path / "negotiations" / f"{buyer.agent_id}_vs_{seller.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    negotiation_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(negotiation_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved negotiation log to: {negotiation_file}")

    # Save final contract
    contract_file = save_path / "contracts" / f"{buyer.agent_id}_vs_{seller.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    contract_file.parent.mkdir(parents=True, exist_ok=True)

    with open(contract_file, 'w') as f:
        json.dump(results["final_contract"], f, indent=2)

    return {
        "agent_a": agent_a.agent_id,
        "agent_b": agent_b.agent_id,
        "buyer": buyer.agent_id,
        "seller": seller.agent_id,
        "buyer_final_budget": results["buyer_final_budget"],
        "seller_final_budget": results["seller_final_budget"],
        "total_actions": len(results["action_log"]),
        "negotiation_file": str(negotiation_file),
        "contract_file": str(contract_file),
        "timestamp": results["timestamp"]
    }


def run_tournament(
    agents: List[LLMAgent],
    baseline_contract: ContractDocument,
    turns: int = 6,
    budget: int = 10,
    save_dir: str = "results"
) -> pd.DataFrame:
    """
    Run a full round-robin tournament.

    Args:
        agents: List of agents to compete
        baseline_contract: Starting contract for all matches
        turns: Number of turns per match
        budget: Budget per agent per match
        save_dir: Directory to save results

    Returns:
        DataFrame with all match results
    """
    matches = create_round_robin_matches(agents)
    results = []

    print(f"\n{'='*70}")
    print(f"TOURNAMENT START: {len(agents)} agents, {len(matches)} matches")
    print(f"{'='*70}\n")

    for match_num, (agent_a, agent_b) in enumerate(matches, 1):
        print(f"\n[Match {match_num}/{len(matches)}]")

        match_result = run_match(
            agent_a=agent_a,
            agent_b=agent_b,
            baseline_contract=baseline_contract,
            turns=turns,
            budget=budget,
            save_dir=save_dir
        )

        results.append(match_result)

        # Save intermediate results
        df = pd.DataFrame(results)
        save_path = Path(save_dir) / "tournaments"
        save_path.mkdir(parents=True, exist_ok=True)
        csv_file = save_path / f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)

    print(f"\n{'='*70}")
    print(f"TOURNAMENT COMPLETE: {len(matches)} matches finished")
    print(f"{'='*70}\n")

    return pd.DataFrame(results)


def print_tournament_summary(results: pd.DataFrame) -> None:
    """Print a summary of tournament results."""
    print("\n" + "="*70)
    print("TOURNAMENT SUMMARY")
    print("="*70 + "\n")

    print(f"Total matches: {len(results)}")
    print(f"Unique agents: {len(set(results['agent_a']).union(set(results['agent_b'])))}")

    print("\nBudget Usage Statistics:")
    print(f"  Buyer avg remaining: {results['buyer_final_budget'].mean():.2f}")
    print(f"  Seller avg remaining: {results['seller_final_budget'].mean():.2f}")

    print("\nAction Statistics:")
    print(f"  Average actions per match: {results['total_actions'].mean():.2f}")
    print(f"  Max actions in a match: {results['total_actions'].max()}")
    print(f"  Min actions in a match: {results['total_actions'].min()}")

    print("\n" + "="*70 + "\n")
