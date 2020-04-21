import numpy
import class_currencypair

class NGARCH(): 
    """
Order of parameters: [alpha, beta, gamma, delta, omega]
    """
    #Function for calulating the volatility, with different epsilon definitions for simulated or actual
    def calculate_h(self, h, i, mode):
        if mode == 'simulated':
            epsilon = self.rand_values[i-1]
        else:
            self.error[i-1] = (self.pair.data[self.read_start + i-1] - self.delta*self.act_h[i-1])/numpy.sqrt(self.act_h[i-1])
            epsilon = self.error[i-1]
        h_i = self.omega + self.alpha * (epsilon + self.gamma * numpy.sqrt(h)) ** 2 + self.beta * h
        #if h_i > 1:
        #    h_i = 1
        return h_i

    #Function for calculating the actual variance (volatility) h_act
    def actual_h(self):
        self.act_h[0] = self.omega
        for i in range(1, len(self.act_h)):
            self.act_h[i] = self.calculate_h(self.act_h[i-1], i, 'actual')

    #Function for simulating the variance (volatility) h_sim
    def simulated_h(self, rand_seed=0):
        self.sim_h[0] = self.omega
        numpy.random.seed(rand_seed)
        self.rand_values = numpy.random.normal(0, 1, self.forecast_steps-1)
        for i in range(1, len(self.sim_h)):
            self.sim_h[i] = self.calculate_h(self.sim_h[i-1], i, 'simulated')

    #Function for transforming the volatility into the log returns R
    def simulated_R(self, rand_seed=10000000):
        numpy.random.seed(rand_seed)
        self.rand_values2 = numpy.random.normal(0, 1, self.forecast_steps)
        for i in range(0, len(self.sim_h)):
            self.sim_R[i] = self.rand_values2[i] * numpy.sqrt(self.sim_h[i]) + self.delta * self.sim_h[i]

    #Function to calculate the negative log-likelihood, which is to be minimised
    def LogL(self, alpha, beta, gamma, delta, omega):
        self.LL = numpy.zeros(len(self.pair.data))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.actual_h()

        for i in range(0, self.read_steps):
            self.LL[i] = 0.5 * (numpy.log(2*numpy.pi) + numpy.log(self.act_h[i]) + self.error[i]**2)
        self.Total_LL = numpy.nansum(self.LL)

        #Penalty term to enforce parameter bounds
        penalty = 0.0
        crit = self.beta + self.alpha * self.gamma ** 2
        if crit > 1:
            penalty = numpy.exp((crit - 1) * 10000) - 1
        elif crit < 0:
            penalty = numpy.exp(-10000 * crit) - 1

        self.weighted_LL = self.Total_LL + penalty
        if self.weighted_LL > 1000000:
            self.weighted_LL = 1000000

        #print([self.alpha, self.beta, self.gamma, self.delta, self.omega])
        #print(weighted_LL)

        return self.weighted_LL

    #Class initialisation
    def __init__(self, pair, params, read_start=0, read_steps='MAX', forecast_start=0, forecast_steps='MAX'):
        self.pair = pair
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.delta = params[3]
        self.omega = params[4]

        self.read_start = read_start
        self.forecast_start = forecast_start
        if read_steps == 'MAX':
            self.read_steps = len(self.pair.data) - self.read_start
        else:
            self.read_steps = read_steps
        if forecast_steps == 'MAX':
            self.forecast_steps = len(self.pair.data)
        else:
            self.forecast_steps = forecast_steps

        self.act_h = numpy.zeros(self.read_steps)
        self.error = numpy.zeros(self.read_steps)
        self.sim_h = numpy.zeros(self.forecast_steps)
        self.sim_R = numpy.zeros(self.forecast_steps)

        self.actual_h()
        self.simulated_h()
        self.simulated_R()