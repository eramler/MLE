import numpy

class NGARCH: 

    #Function for simulating the variance (volatility) h
    def variance_h(self):
        for i in range(1, len(self.h)):
            rand = numpy.random.normal(0, 1, 1)
            self.h[i] = self.omega + self.beta * self.h[i-1] + self.delta * (rand - self.gamma * numpy.sqrt(self.h[i-1])) ** 2
    
    #Function for transforming the volatility into the log returns R
    def returns_transform(self):
        for i in range(1, len(self.h)):
            rand = numpy.random.normal(0, 1, 1)
            self.R[i] = rand * numpy.sqrt(self.h[i]) + self.alpha * self.h[i]

    #Class initialisation
    def __init__(self, name, params, steps):
        self.name = name
        self.steps = steps
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.delta = params[3]
        self.omega = params[4]

        self.h = numpy.zeros(self.steps)
        self.R = numpy.zeros(self.steps)

        self.variance_h()
        self.returns_transform()