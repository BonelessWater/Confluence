from dataclasses import dataclass, field
from typing import Dict, Optional
import time

@dataclass
class ExchangeQuote:
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    last_update: float = 0.0

class MultiExchangeBook:
    def __init__(self):
        self.exchanges: Dict[str, ExchangeQuote] = {}

    def update_quote(self, exchange: str, side: str, price: float, size: int, timestamp: float):
        if exchange not in self.exchanges:
            self.exchanges[exchange] = ExchangeQuote()
        q = self.exchanges[exchange]

        if side == "BID":
            q.bid = price
            q.bid_size = size
        elif side == "ASK":
            q.ask = price
            q.ask_size = size

        q.last_update = timestamp

    def get_valid_exchanges(self):
        return [q for q in self.exchanges.values() if q.bid > 0 and q.ask > 0]
