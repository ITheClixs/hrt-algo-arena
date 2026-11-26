"""Task definitions for the HRT challenge variants seen so far.

Change these (or make a new TaskSpec) the minute the organisers explain the rules on the day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Where you trade, how long you hold, and how the official scorer computes Sharpe."""

    name: str
    halfway_bar: int  # you trade at the close of this bar
    exit_bar: int  # you close the position at the close of this bar
    annualization: float  # Sharpe multiplier, e.g. sqrt(252) or 16
    ddof: int = 1  # std ddof the official scorer uses (pandas = 1, numpy = 0)
    demean_returns: bool = False  # True if returns are market-adjusted (demeaned across sessions)
    id_col: str = "session"  # id column name in the upload file
    pred_col: str = "pred"  # prediction column name in the upload file

    @property
    def horizon(self) -> int:
        return self.exit_bar - self.halfway_bar


# Algo Arena on HRT's "arcade" platform (Autumn 2026 student events): 10k in-sample sessions,
# 10k out-of-sample sessions, predict the market-adjusted close(49) -> close(59) return,
# 10 uploads, live leaderboard. Upload columns: headline_id,pred (headline_id = session number).
ARCADE_ALGO_ARENA = TaskSpec(
    name="arcade-algo-arena",
    halfway_bar=49,
    exit_bar=59,
    annualization=math.sqrt(252),
    ddof=1,
    demean_returns=True,
    id_col="headline_id",
    pred_col="pred",
)

# ETH Zurich Datathon 2026 HRT challenge (Kaggle): 1k train sessions, hold from close(49) to
# close(99), Sharpe * 16, upload columns session,target_position.
DATATHON_2026 = TaskSpec(
    name="eth-datathon-2026",
    halfway_bar=49,
    exit_bar=99,
    annualization=16.0,
    ddof=0,
    demean_returns=False,
    id_col="session",
    pred_col="target_position",
)
