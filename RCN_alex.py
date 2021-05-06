import numpy as np
import matplotlib.pyplot as plt


class RCN_binomial():

    def __init__(self, interest_rate, period_length, initial_price, dividend_yield, up_factor, down_factor,
                 maturity, annualized_coupon, exercise_price, barrier_level=None, _callable=False):
        # binomial model parameters
        self.r = interest_rate
        self.Delta = period_length
        self.i0 = initial_price
        self.delta = dividend_yield
        self.U = up_factor
        self.D = down_factor

        self.gamma = np.exp(-self.r * self.Delta)
        self.q = (1 / self.gamma - self.D) / (self.U - self.D)

        # RCN characteristics
        self.beta = barrier_level
        self.callable = _callable
        self.T = maturity
        self.c = annualized_coupon
        self.alpha = exercise_price

    def price_RCN(self):
        if self.q <= 0 or self.q >= 1:
            raise Exception("No arbitrage condition on U, D and r not satisfied.")
        if not self.beta:
            # simple RCN: can use Markov property
            I = np.full((self.T + 1, self.T + 1), np.nan)
            I[0, 0] = self.i0
            for t in range(1, self.T + 1):
                for i in range(t + 1):
                    if i == 0:
                        I[i, t] = I[i, t - 1] * self.U * np.exp(-self.delta * self.Delta)
                    else:
                        I[i, t] = I[i - 1, t - 1] * self.D * np.exp(-self.delta * self.Delta)
            H = self.alpha * self.i0 - I
            P = np.full((self.T + 1, self.T + 1), np.nan)
            P[:, self.T] = self.c * self.Delta + 1 - np.maximum(H[:, self.T], 0) / self.i0
            if not self.callable:
                for t in range(self.T - 1, -1, -1):
                    for i in range(t + 1):
                        if t != 0:
                            P[i, t] = (
                                    self.gamma * (self.q * P[i, t + 1] + (1 - self.q) * P[i + 1, t + 1])
                                    + self.c * self.Delta
                            )
                        else:
                            P[i, t] = self.gamma * (self.q * P[i, t + 1] + (1 - self.q) * P[i + 1, t + 1])
            else:
                for t in range(self.T - 1, -1, -1):
                    for i in range(t + 1):
                        if t != 0:
                            P[i, t] = np.minimum(
                                self.gamma * (self.q * P[i, t + 1] + (1 - self.q) * P[i + 1, t + 1]),
                                1
                            ) + self.c * self.Delta
                        else:
                            P[i, t] = self.gamma * (self.q * P[i, t + 1] + (1 - self.q) * P[i + 1, t + 1])
        else:
            # Barrier RCN: can't use Markov property
            I = np.full((2 ** self.T, self.T + 1), np.nan)
            B = np.full((2 ** self.T, self.T + 1), np.nan)  # matrix to track down-and-in activation
            I[0, 0] = self.i0
            B[0, 0] = False
            for t in range(1, self.T + 1):
                for i in range(2 ** t):
                    if i % 2 == 0:
                        I[i, t] = I[i // 2, t - 1] * self.U * np.exp(-self.delta * self.Delta)
                    else:
                        I[i, t] = I[i // 2, t - 1] * self.D * np.exp(-self.delta * self.Delta)
                    if B[i // 2, t - 1]:  # check if barrier was reached previously and propagate
                        B[i, t] = True
                    else:  # if not, check now
                        B[i, t] = I[i, t] <= self.beta * self.i0
            H = np.where(B, self.alpha * self.i0 - I, 0)
            P = np.full((2 ** self.T, self.T + 1), np.nan)
            P[:, self.T] = self.c * self.Delta + 1 - np.maximum(H[:, self.T], 0) / self.i0
            if not self.callable:
                for t in range(self.T - 1, -1, -1):
                    for i in range(2 ** t):
                        if t != 0:
                            P[i, t] = (
                                    self.gamma * (self.q * P[2 * i, t + 1] + (1 - self.q) * P[2 * i + 1, t + 1])
                                    + self.c * self.Delta
                            )
                        else:
                            P[i, t] = self.gamma * (self.q * P[2 * i, t + 1] + (1 - self.q) * P[2 * i + 1, t + 1])
            else:
                for t in range(self.T - 1, -1, -1):
                    for i in range(2 ** t):
                        if t != 0:
                            P[i, t] = np.minimum(
                                self.gamma * (self.q * P[2 * i, t + 1] + (1 - self.q) * P[2 * i + 1, t + 1]),
                                1
                            ) + self.c * self.Delta
                        else:
                            P[i, t] = self.gamma * (self.q * P[2 * i, t + 1] + (1 - self.q) * P[2 * i + 1, t + 1])
        return P[0, 0]


class Binomial_calibrator():

    def __init__(self, interest_rate, period_length, initial_price, dividend_yield, up_factors, q, maturity, strikes,
                 call_prices):
        # binomial model parameters
        self.r = interest_rate
        self.Delta = period_length
        self.i0 = initial_price
        self.delta = dividend_yield
        self.Us = up_factors
        self.q = q
        self.T = maturity
        self.gamma = np.exp(-self.r * self.Delta)

        # known prices
        self.strikes = strikes
        self.call_prices = call_prices

    def price_call(self, U, D, strike):
        "Prices european call options"
        # we can use Markov property
        I = np.full((self.T + 1, self.T + 1), np.nan)
        I[0, 0] = self.i0
        for t in range(1, self.T + 1):
            for i in range(t + 1):
                if i == 0:
                    I[i, t] = I[i, t - 1] * U * np.exp(-self.delta * self.Delta)
                else:
                    I[i, t] = I[i - 1, t - 1] * D * np.exp(-self.delta * self.Delta)
        P = np.full((self.T + 1, self.T + 1), np.nan)
        P[:, self.T] = np.maximum(I[:, self.T] - strike, 0)
        for t in range(self.T - 1, -1, -1):
            for i in range(t + 1):
                P[i, t] = self.gamma * (self.q * P[i, t + 1] + (1 - self.q) * P[i + 1, t + 1])
        return P[0, 0]

    def compute_mse(self, U):
        D = (np.exp(self.Delta * self.r) - self.q * U) / (1 - self.q)
        model_call_prices = []
        for strike in self.strikes:
            model_call_prices.append(self.price_call(U, D, strike))
        return np.mean(np.square(self.call_prices - np.array(model_call_prices)))

    def calibrate(self):
        mses = []
        for U in self.Us:
            mses.append(self.compute_mse(U))
        best_U = self.Us[np.argmin(mses)]
        best_D = (np.exp(self.Delta * self.r) - self.q * best_U) / (1 - self.q)
        print(
            """
            Best parameters:
                - up factor: {:.5f}
                - down factor: {:.5f}
            """.format(best_U, best_D)
        )

        fig, ax = plt.subplots()
        ax.plot(self.Us, mses)
        ax.set(
            xlabel="U",
            ylabel="Mean squared error",
            title="Results of parameter calibration"
        )
        plt.show()
        return best_U, best_D
