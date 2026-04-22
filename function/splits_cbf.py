import json
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Fold:
    fold: int
    train_players: List[str]
    test_players: List[str]


def load_player_folds(path: str) -> Tuple[List[str], List[Fold]]:
    """
    Load a fixed player-based split definition JSON.
    Returns (players, folds).
    """
    with open(path, "r") as f:
        obj = json.load(f)
    players = obj["players"]
    folds = [
        Fold(
            fold=fd["fold"],
            train_players=fd["train_players"],
            test_players=fd["test_players"],
        )
        for fd in obj["folds"]
    ]
    return players, folds

