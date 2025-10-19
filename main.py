"""
Main entry point for the AI Contract Negotiation Competition.

This script loads models, runs the tournament, and saves results.
"""

import argparse
from datetime import datetime
from contracts import ContractDocument
from agents import LLMAgent
from tournament import run_tournament, print_tournament_summary
import config


def main():
    """Run the contract negotiation tournament."""

    parser = argparse.ArgumentParser(
        description="AI Contract Negotiation Competition V1"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to use (default: all models in config)"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=config.NEGOTIATION_CONFIG["total_turns"],
        help=f"Number of turns per negotiation (default: {config.NEGOTIATION_CONFIG['total_turns']})"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=config.NEGOTIATION_CONFIG["budget_per_agent"],
        help=f"Edit budget per agent (default: {config.NEGOTIATION_CONFIG['budget_per_agent']})"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only use first 2 models"
    )

    args = parser.parse_args()

    # Print configuration
    config.print_config()

    # Determine which models to use
    if args.test:
        print("\n*** TEST MODE: Using small test models ***\n")
        model_names = config.TEST_MODELS
    elif args.models:
        model_names = args.models
    else:
        model_names = config.MODELS

    print(f"Loading {len(model_names)} models...\n")

    # Load baseline contract
    print(f"Loading baseline contract from {config.PATHS['baseline_contract']}...")
    baseline_contract = ContractDocument.from_json_file(
        str(config.PATHS['baseline_contract'])
    )
    print(f"Baseline contract has {len(baseline_contract.list_articles())} articles\n")

    # Initialize agents
    agents = []
    for model_name in model_names:
        try:
            agent = LLMAgent(
                model_name=model_name,
                role="neutral",  # Will be assigned buyer/seller in each match
                agent_id=model_name.split("/")[-1]  # Short name
            )
            agents.append(agent)
        except Exception as e:
            print(f"ERROR loading {model_name}: {e}")
            print(f"Skipping this model...\n")

    if len(agents) < 2:
        print("ERROR: Need at least 2 agents to run tournament")
        return

    print(f"\nSuccessfully loaded {len(agents)} agents\n")

    # Run tournament
    start_time = datetime.now()
    print(f"Tournament start time: {start_time}\n")

    results_df = run_tournament(
        agents=agents,
        baseline_contract=baseline_contract,
        turns=args.turns,
        budget=args.budget,
        save_dir=str(config.PATHS['results_dir'])
    )

    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print_tournament_summary(results_df)

    print(f"Tournament duration: {duration}")
    print(f"Average time per match: {duration / len(results_df)}")

    # Save final results
    final_csv = config.PATHS['tournaments_dir'] / f"tournament_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(final_csv, index=False)
    print(f"\nFinal results saved to: {final_csv}")

    print("\n" + "="*70)
    print("TOURNAMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
