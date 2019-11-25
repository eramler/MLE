import numpy
import class_currencypair

class NGARCH(): 
    """
Order of parameters: [alpha, beta, gamma, delta, omega]
    """
    #Function for calulating the volatility, either simulated or actual
    def calculate_h(self, h, i, mode):
        if mode == 'simulated':
            epsilon = numpy.random.normal(0, 1, 1)
        else:
            #########################
            self.error[i-1] = (self.pair.data[i-1] - self.alpha*self.act_h[i-1])/numpy.sqrt(self.act_h[i-1])
            epsilon = self.error[i-1]
        h_i = self.omega + self.beta * h + self.delta * (epsilon - self.gamma * numpy.sqrt(h)) ** 2
        return h_i

    #Function for calculating the actual variance (volatility) h_act
    def actual_h(self):
        self.act_h[0] = self.omega
        for i in range(1, len(self.act_h)-1):
            self.act_h[i] = self.calculate_h(self.act_h[i-1], i, 'actual')

    #Function for simulating the variance (volatility) h_sim
    def simulated_h(self):
        for i in range(1, len(self.sim_h)-1):
            self.sim_h[i] = self.calculate_h(self.sim_h[i-1], i, 'simulated')
    
    #Function for transforming the volatility into the log returns R
    def simulated_R(self):
        for i in range(0, len(self.sim_h)-1):
            rand = numpy.random.normal(0, 1, 1)
            self.sim_R[i] = rand * numpy.sqrt(self.sim_h[i]) + self.alpha * self.sim_h[i]

    #Function to calculate the log-likelihood from the simulated and actual data
    def LogL(self, params):
        #self.alpha = params[0]
        #self.beta = params[1]
        #self.gamma = params[2]
        #self.delta = params[3]
        #self.omega = params[4]
        for i in range(0, len(self.pair.data)-1):
            #Minus sign, to allow minimise negative function instead of maximising regular function
            self.LL = self.LL - 0.5 * (numpy.log(2*numpy.pi) + numpy.log(self.act_h[i]) + self.error[i]**2)
        print(self.LL)
        return self.LL

    #Class initialisation
    def __init__(self, pair, params, steps):
        self.pair = pair
        self.steps = steps
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.delta = params[3]
        self.omega = params[4]

        self.LL = 0.0

        self.act_h = numpy.zeros(len(self.pair.data))
        self.error = numpy.zeros(len(self.pair.data))
        self.sim_h = numpy.zeros(self.steps)
        self.sim_R = numpy.zeros(self.steps)

        self.simulated_h()
        self.simulated_R()
        self.actual_h()
        self.LogL(params)
        #params[0] = 0.02
        #self.LogL(params)