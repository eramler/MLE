import numpy
import class_currencypair

class NGARCH(): 
    """
Order of parameters: [alpha, beta, gamma, delta, omega]
    """
    #Function for calulating the volatility, with different epsilon definitions for simulated or actual
    def calculate_h(self, h, i, mode):
        if mode == 'simulated':
            epsilon = numpy.random.normal(0, 1, 1)
        else:
            self.error[i-1] = (self.pair.data[i-1] - self.delta*self.act_h[i-1])/numpy.sqrt(self.act_h[i-1])
            epsilon = self.error[i-1]
        h_i = self.omega + self.alpha * (epsilon + self.gamma * numpy.sqrt(h)) ** 2 + self.beta * h
        return h_i

    #Function for calculating the actual variance (volatility) h_act
    def actual_h(self):
        self.act_h[0] = self.omega
        for i in range(1, len(self.act_h)):
            self.act_h[i] = self.calculate_h(self.act_h[i-1], i, 'actual')

    #Function for simulating the variance (volatility) h_sim
    def simulated_h(self):
        self.sim_h[0] = self.omega
        for i in range(1, len(self.sim_h)):
            self.sim_h[i] = self.calculate_h(self.sim_h[i-1], i, 'simulated')

    #Function for transforming the volatility into the log returns R
    def simulated_R(self):
        for i in range(0, len(self.sim_h)):
            rand = numpy.random.normal(0, 1, 1)
            self.sim_R[i] = rand * numpy.sqrt(self.sim_h[i]) + self.delta * self.sim_h[i]

    #Function to calculate the negative log-likelihood, which is to be minimised
    def LogL(self, alpha, beta, gamma, delta, omega):
        self.LL = numpy.zeros(len(self.pair.data))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.actual_h()

        for i in range(0, len(self.pair.data)):
            self.LL[i] = 0.5 * (numpy.log(2*numpy.pi) + numpy.log(self.act_h[i]) + self.error[i]**2)
        Total_LL = numpy.nansum(self.LL)

        #Penalty term to enforce parameter bounds
        penalty = 0.0
        crit = self.beta + self.alpha * self.gamma ** 2
        if crit > 1:
            penalty = numpy.exp((crit - 1) * 10000) - 1
        elif crit < 0:
            penalty = numpy.exp(-10000 * crit) - 1

        weighted_LL = Total_LL + penalty
        if weighted_LL > 1000000:
            weighted_LL = 1000000

        #print([self.alpha, self.beta, self.gamma, self.delta, self.omega])
        #print(weighted_LL)

        return weighted_LL

    #Class initialisation
    def __init__(self, pair, params, steps):
        self.pair = pair
        self.steps = steps
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.delta = params[3]
        self.omega = params[4]

        self.act_h = numpy.zeros(len(self.pair.data))
        self.error = numpy.zeros(len(self.pair.data))
        self.sim_h = numpy.zeros(self.steps)
        self.sim_R = numpy.zeros(self.steps)

        self.simulated_h()
        self.simulated_R()
        self.actual_h()