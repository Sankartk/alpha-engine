"""
Universe: defines the tradeable symbol set with point-in-time membership.

Avoids survivorship bias by supporting historical index membership snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA",
    "XOM", "CVX", "LLY", "ABBV", "PFE",
]

LIQUID_ETFS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "GLD", "XLF", "XLK", "XLE",
]


@dataclass
class Universe:
    """
    Point-in-time symbol universe.

    Members can be filtered by date to simulate index reconstitution.
    """

    name: str
    symbols: list[str] = field(default_factory=list)
    _membership: dict[str, list[tuple[datetime, datetime | None]]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def sp500_sample(cls) -> Universe:
        return cls(name="sp500_sample", symbols=SP500_SAMPLE)

    @classmethod
    def liquid_etfs(cls) -> Universe:
        return cls(name="liquid_etfs", symbols=LIQUID_ETFS)

    def members_at(self, dt: datetime) -> list[str]:
        """Return symbols that were members at dt. Defaults to all if no membership data."""
        if not self._membership:
            return self.symbols
        return [
            sym for sym, windows in self._membership.items()
            if any(start <= dt and (end is None or dt <= end) for start, end in windows)
        ]

    def add_membership(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
    ) -> None:
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        self._membership.setdefault(symbol, []).append((start, end))

    def __len__(self) -> int:
        return len(self.symbols)

    def __repr__(self) -> str:
        return f"Universe({self.name!r}, {len(self.symbols)} symbols)"
