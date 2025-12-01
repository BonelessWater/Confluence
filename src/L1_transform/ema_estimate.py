class EMAEfficientPrice:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = None

    def update(self, observed_price: float):
        if observed_price is None:
            return self.value

        if self.value is None:
            self.value = observed_price
        else:
            self.value = self.alpha * observed_price + (1 - self.alpha) * self.value

        return self.value
