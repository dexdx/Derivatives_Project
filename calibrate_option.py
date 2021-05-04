import numpy as np

class calibrate_option():

    def __init__(self, interest_rate, period_length, initial_price, dividend_yield, up_factor, emm, maturity, strike):
        # binomial model parameters
        self.r = interest_rate
        self.Delta = period_length
        self.i0 = initial_price
        self.delta = dividend_yield
        self.U = up_factor
        self.q = emm
        self.D = (np.exp(self.Delta*self.r)-self.q*self.U)/(1-self.q)
        self.T = maturity
        self.strike = strike
        
        self.gamma = np.exp(-self.r*self.Delta)
        
    
    def terminal_payoffs(self, put=True):
        t = np.arange(1, self.T+1, 1)
        # with Markov property
        P = np.zeros((self.T+1, self.T+1))
        P[0,0] = self.i0
        for j in t:
            for i in range(j+1):
                if i == 0:
                    P[i,j] = P[i,j-1] * self.U * (1 - self.delta*self.Delta)
                else:
                    P[i,j] = P[i-1,j-1] * self.D * (1 - self.delta*self.Delta)
        
        H = np.maximum(self.strike - P[:,self.T], 0)
        if not put:
            H = np.maximum(P[:,self.T] - self.strike, 0)
        return H
    
    def price_option(self, put=True):
        if self.q <= 0 or self.q >= 1:
            return
        t = np.arange(self.T-1,-1,-1)
        # with Markov propoerty
        P = np.zeros((self.T+1, self.T+1))
        P[:,self.T] = self.terminal_payoffs(put)
        for j in t:
            for i in range(j+1):
                P[i,j] = self.gamma*( self.q*P[i, j+1] + (1-self.q)*P[i+1, j+1] )
        return P[0,0]
