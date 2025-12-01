import math
from .per_exchange_state import MultiExchangeBook

def compute_consensus_mid(book: MultiExchangeBook, now: float, lambda_decay=2.0):
    exs = book.get_valid_exchanges()
    if not exs:
        return None

    weights = []
    mids = []

    for q in exs:
        spread = max(q.ask - q.bid, 1e-9)
        size = q.bid_size + q.ask_size
        age = now - q.last_update
        w = (size / spread) * math.exp(-lambda_decay * age)

        weights.append(w)
        mids.append((q.bid + q.ask) / 2)

    total_w = sum(weights)
    return sum(w * m for w, m in zip(weights, mids)) / total_w
