"""
Contract representation and manipulation for AI negotiation system.

This module provides the core contract data structures for the negotiation
competition. Contracts are represented as JSON-compatible dictionaries with
named articles.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Article:
    """Represents a single article in a contract."""
    content: str
    last_modified_by: str
    modification_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert article to dictionary."""
        return {
            "content": self.content,
            "last_modified_by": self.last_modified_by,
            "modification_timestamp": self.modification_timestamp
        }


class ContractDocument:
    """
    Represents a contract with multiple articles.

    Contracts consist of named articles (e.g., 'price_terms', 'delivery_terms')
    and maintain a history of all modifications.
    """

    def __init__(self, articles: Dict[str, Article], metadata: Optional[dict] = None):
        """
        Initialize a contract.

        Args:
            articles: Dictionary mapping article names to Article objects
            metadata: Optional metadata (parties, subject, etc.)
        """
        self.articles = articles
        self.metadata = metadata or {
            "parties": {
                "buyer": "West Manufacturing Inc.",
                "seller": "Square Machines Inc."
            },
            "subject": "Sale of 100 machines"
        }
        self.history: List[dict] = []

    def to_dict(self) -> dict:
        """Convert contract to JSON-serializable dictionary."""
        return {
            "metadata": self.metadata,
            "articles": {
                name: article.to_dict()
                for name, article in self.articles.items()
            }
        }

    def apply_action(self, action: dict, agent_id: str) -> None:
        """
        Apply a negotiation action to modify the contract.

        Args:
            action: Dictionary with action_type and parameters
            agent_id: ID of the agent making the change
        """
        action_type = action.get("action_type")

        if action_type == "EDIT_ARTICLE":
            article_name = action["article_name"]
            new_content = action["content"]
            self.articles[article_name] = Article(
                content=new_content,
                last_modified_by=agent_id
            )

        elif action_type == "ADD_ARTICLE":
            article_name = action["article_name"]
            content = action["content"]
            self.articles[article_name] = Article(
                content=content,
                last_modified_by=agent_id
            )

        elif action_type == "REMOVE_ARTICLE":
            article_name = action["article_name"]
            if article_name in self.articles:
                del self.articles[article_name]

        elif action_type == "PASS":
            # No changes to contract
            pass

        # Record action in history
        self.history.append({
            "agent": agent_id,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })

    def get_article(self, name: str) -> Optional[Article]:
        """Get an article by name."""
        return self.articles.get(name)

    def list_articles(self) -> List[str]:
        """Get list of all article names."""
        return list(self.articles.keys())

    @classmethod
    def from_dict(cls, data: dict) -> 'ContractDocument':
        """Create ContractDocument from dictionary."""
        articles = {
            name: Article(
                content=article_data["content"],
                last_modified_by=article_data["last_modified_by"],
                modification_timestamp=article_data.get(
                    "modification_timestamp",
                    datetime.now().isoformat()
                )
            )
            for name, article_data in data.get("articles", {}).items()
        }
        return cls(articles=articles, metadata=data.get("metadata"))

    @classmethod
    def from_json_file(cls, filepath: str) -> 'ContractDocument':
        """Load contract from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, filepath: str) -> None:
        """Save contract to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Action cost mapping
ACTION_COSTS = {
    "EDIT_ARTICLE": 1,
    "ADD_ARTICLE": 2,
    "REMOVE_ARTICLE": 2,
    "PASS": 0
}


def get_action_cost(action: dict) -> int:
    """Get the budget cost for an action."""
    action_type = action.get("action_type", "PASS")
    return ACTION_COSTS.get(action_type, 0)
