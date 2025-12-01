import math
from .per_exchange_state import MultiExchangeBook

def compute_consensus_micro(book: MultiExchangeBook, now: float, lambda_decay=2.0):
    exs = book.get_valid_exchanges()
    if not exs:
        return None

    weights = []
    micros = []

    for q in exs:
        size = q.bid_size + q.ask_size
        if size == 0:
            continue

        micro = (q.ask_size * q.bid + q.bid_size * q.ask) / size

        spread = max(q.ask - q.bid, 1e-9)
        age = now - q.last_update
        w = (size / spread) * math.exp(-lambda_decay * age)

        weights.append(w)
        micros.append(micro)

    total_w = sum(weights)
    return sum(w * m for w, m in zip(weights, micros)) / total_w
