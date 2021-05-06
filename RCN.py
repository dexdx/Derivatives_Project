import numpy as np

class RCN_binomial():

    def __init__(self, interest_rate, period_length, initial_price, dividend_yield, up_factor, down_factor,
                 payment_dates, annualized_coupon, exercise_price, Simple=True, Callable=False, barrier_level=None):
        # binomial model parameters
        self.r = interest_rate
        self.Delta = period_length
        self.i0 = initial_price
        self.delta = dividend_yield
        self.U = up_factor
        self.D = down_factor
        
        self.gamma = np.exp(-self.r*self.Delta)
        self.q = (1/self.gamma - self.D)/(self.U - self.D)
        
        # RCN characteristics
        self.simple = Simple
        self.callable = Callable
        self.dates = payment_dates
        self.T = payment_dates[len(payment_dates)-1]
        self.c = annualized_coupon
        self.alpha = exercise_price
        self.beta = barrier_level   
        self.barrier = None
        
    def set_barrier(self):
        if self.beta is not None:
            self.barrier = self.beta*self.i0
    
    def terminal_payoffs(self, put=True):
        t = np.arange(1, self.T+1, 1)
        if self.simple:
            # with Markov property
            P = np.zeros((self.T+1, self.T+1))
            P[0,0] = self.i0
            for j in t:
                for i in range(j+1):
                    if i == 0:
                        P[i,j] = P[i,j-1] * self.U * (1 - self.delta*self.Delta)
                    else:
                        P[i,j] = P[i-1,j-1] * self.D * (1 - self.delta*self.Delta)
        else:
            # can't use Markov property with barrier
            P = np.zeros((2**self.T, self.T+1))
            B = np.zeros((2**self.T, self.T+1), dtype=bool) # matrix of price if BELOW barrier (init False)
            P[0,0] = self.i0
            for j in t:
                for i in range(2**j):
                    if i%2 == 0:
                        P[i,j] = P[int(i/2),j-1] * self.U * (1 - self.delta*self.Delta)
                    else:
                        P[i,j] = P[int((i-1)/2),j-1] * self.D * (1 - self.delta*self.Delta)
                    if j < self.T:
                        if not B[i,j]: # if not already True, check price and barrier
                            if P[i,j] <= self.beta*self.i0: # change to True and "propagate" True to next direct nodes
                                B[i,j] = True
                                B[2*i,j+1] = True
                                B[2*i+1,j+1] = True
                        else: # propagate to next direct nodes
                            B[2*i,j+1] = True
                            B[2*i+1,j+1] = True
        
        H = np.maximum(self.alpha*self.i0 - P[:,self.T], 0)
        if not self.simple:
            H = H * B[:,self.T] # payoff only if price was below barrier at some point in the path leading to terminal node
        if not put:
            H = np.maximum(P[:,self.T] - self.alpha*self.i0, 0)
        return H
    
    def price_option(self, put=True):
        if self.q <= 0 or self.q >= 1:
            return
        t = np.arange(self.T-1,-1,-1)
        if self.simple:
            # with Markov propoerty
            P = np.zeros((self.T+1, self.T+1))
            P[:,self.T] = self.terminal_payoffs(put)
            for j in t:
                for i in range(j+1):
                    P[i,j] = self.gamma*( self.q*P[i, j+1] + (1-self.q)*P[i+1, j+1] )
            return P[0,0]
        else:
            # can't use Markov property with barrier
            P = np.zeros((2**self.T, self.T+1))
            P[:,self.T] = self.terminal_payoffs(put)
            for j in t:
                for i in range(2**j):
                    P[i,j] = self.gamma*( self.q*P[2*i, j+1] + (1-self.q)*P[2*i+1, j+1] )
            return P[0,0]
    
    def price_bonds(self):
        pv_bonds = 0
        for i in self.dates:
            pv_bonds += self.Delta*self.c*np.exp(-self.r*self.Delta*i)
        pv_bonds += np.exp(-self.r*self.Delta*self.T)
        return pv_bonds
    
    def price_RCN(self):
        replicating_initial_cashflow = self.price_option()/self.i0 - self.price_bonds()
        return -replicating_initial_cashflow