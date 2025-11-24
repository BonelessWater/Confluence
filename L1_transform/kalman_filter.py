class KalmanEfficientPrice:
    def __init__(self, process_var=0.02, obs_var=0.5):
        self.p = None           # posterior mean
        self.P = None           # posterior covariance
        self.Q = process_var    # process variance
        self.R = obs_var        # observation variance

    def update(self, X):
        if X is None:
            return self.p

        # Initialization
        if self.p is None:
            self.p = X
            self.P = 1.0
            return self.p

        # Predict
        p_pred = self.p
        P_pred = self.P + self.Q

        # Kalman gain
        K = P_pred / (P_pred + self.R)

        # Update
        self.p = p_pred + K * (X - p_pred)
        self.P = (1 - K) * P_pred

        return self.p
