import numpy
import class_currencypair

class NGARCH(class_currencypair.CurrencyPair): 
    """
Order of parameters: [alpha, beta, gamma, delta, omega]
    """

    #Function for simulating the variance (volatility) h
    def simulated_h(self):
        for i in range(0, len(self.h)):
            rand = numpy.random.normal(0, 1, 1)
            self.h[i] = self.omega + self.beta * self.h[i-1] + self.delta * (rand - self.gamma * numpy.sqrt(self.h[i-1])) ** 2
    
    #Function for transforming the volatility into the log returns R
    def returns_transform(self):
        for i in range(1, len(self.h)):
            rand = numpy.random.normal(0, 1, 1)
            self.R[i] = rand * numpy.sqrt(self.h[i]) + self.alpha * self.h[i]

    #Function to calculate the log-likelihood from the simulated and actual data
    def loglikelihood(self):
        for i in range(0, len(self.data)-1):
            self.error[i] = (self.data[i] - self.alpha * self.h[i]) / numpy.sqrt(self.h[i])
            self.LL = self.LL + 0.5 * (numpy.log(2*numpy.pi) + numpy.log(self.h[i]) + self.error[i]**2)

    #Class initialisation
    def __init__(self, name, params, steps):
        class_currencypair.CurrencyPair.__init__(self, name)
        self.name = name
        self.steps = steps
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.delta = params[3]
        self.omega = params[4]

        self.LL = 0.0
        self.h = numpy.zeros(self.steps)
        self.R = numpy.zeros(self.steps)
        self.error = numpy.zeros(len(self.data))

        self.simulated_h()
        self.returns_transform()
        self.loglikelihood()