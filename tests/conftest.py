from __future__ import annotations

import numpy as np
import pytest

from arena.config import ARCADE_ALGO_ARENA
from arena.data import Sessions, forward_returns, from_long
from arena.models import ensure_keys
from arena.synth import SyntheticMarket, make_synthetic, write_arcade_files


@pytest.fixture(scope="session")
def market() -> SyntheticMarket:
    return make_synthetic(n_sessions=400, seed=1)


@pytest.fixture(scope="session")
def sessions(market: SyntheticMarket) -> Sessions:
    return from_long(market.bars, market.headlines)


@pytest.fixture(scope="session")
def keyed(sessions: Sessions) -> Sessions:
    return ensure_keys(sessions)


@pytest.fixture(scope="session")
def y(sessions: Sessions) -> np.ndarray:
    return forward_returns(sessions, ARCADE_ALGO_ARENA)


@pytest.fixture()
def arcade_dir(market: SyntheticMarket, tmp_path):
    return write_arcade_files(market, tmp_path / "arcade")


@pytest.fixture(scope="session")
def extra_market() -> SyntheticMarket:
    """Labelled sessions with ids 0..99, disjoint from `market` (ids 1000+), like the datathon train set."""
    return make_synthetic(n_sessions=100, seed=2, start_id=0)


@pytest.fixture()
def datathon_dir(extra_market: SyntheticMarket, tmp_path):
    """`extra_market` written the way the ETH Datathon shipped its train set: seen bars 0-49, unseen 50-99."""
    out = tmp_path / "datathon"
    out.mkdir()
    bars, hl = extra_market.bars, extra_market.headlines[["session", "bar_ix", "headline"]]
    bars[bars["bar_ix"] <= 49].to_parquet(out / "bars_seen_train.parquet", index=False)
    bars[bars["bar_ix"] > 49].to_parquet(out / "bars_unseen_train.parquet", index=False)
    hl[hl["bar_ix"] <= 49].to_parquet(out / "headlines_seen_train.parquet", index=False)
    hl[hl["bar_ix"] > 49].to_parquet(out / "headlines_unseen_train.parquet", index=False)
    return out
