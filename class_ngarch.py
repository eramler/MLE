import numpy
import class_currencypair
import sys
import math

class NGARCH(): 
    """
Order of parameters: [alpha, beta, gamma, delta, omega]
    """
    #Function for calulating the volatility, either simulated or actual
    def calculate_h(self, h, i, mode):
        if mode == 'simulated':
            epsilon = numpy.random.normal(0, 1, 1)
        else:
            if i < 1:
                print('Error!!! Index less than 1 when trying to calculate the volatility!!!')
            else:
                self.error[i-1] = (self.pair.data[i-1] - self.alpha*self.act_h[i-1])/numpy.sqrt(self.act_h[i-1])
                epsilon = self.error[i-1]
        h_i = self.omega + self.beta * h + self.delta * (epsilon - self.gamma * numpy.sqrt(h)) ** 2

        #if h_i < 0:
        #        print(i)
        #        sys.exit('volatility calculated to be less than one!!')

        #print(h_i)

        return h_i

    #Function for calculating the actual variance (volatility) h_act
    def actual_h(self):
        self.act_h[0] = self.omega
        for i in range(1, len(self.act_h)):
            #if self.act_h[i-1] >= 1:
            #    self.act_h[i-1] = 0.99
            #if self.act_h[i-1] <= 0:
            #    self.act_h[i-1] = 1e-5
            self.act_h[i] = self.calculate_h(self.act_h[i-1], i, 'actual')



    #Function for simulating the variance (volatility) h_sim
    def simulated_h(self):
        self.sim_h[0] = self.omega
        for i in range(1, len(self.sim_h)):
            #if self.sim_h[i-1] >= 1:
            #    self.sim_h[i-1] = 0.99
            #if self.sim_h[i-1] <= 0:
            #    self.sim_h[i-1] = 1e-5
            self.sim_h[i] = self.calculate_h(self.sim_h[i-1], i, 'simulated')
            
    
    #Function for transforming the volatility into the log returns R
    def simulated_R(self):
        for i in range(0, len(self.sim_h)):
            rand = numpy.random.normal(0, 1, 1)
            self.sim_R[i] = rand * numpy.sqrt(self.sim_h[i]) + self.alpha * self.sim_h[i]

    #Function to calculate the log-likelihood from the simulated and actual data
    def LogL(self, alpha, beta, gamma, delta, omega):
        self.attempt = self.attempt + 1
        self.LL = numpy.zeros(len(self.pair.data))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.actual_h()
        for i in range(0, len(self.pair.data)):
            #Minus sign, to allow minimise negative function instead of maximising regular function
            self.LL[i] = 0.5 * (numpy.log(2*numpy.pi) + numpy.log(self.act_h[i]) + self.error[i]**2)

            #print('iteration:' + str(i))
            #if math.isnan(self.LL[i]) == True:
            #    sys.exit('LL is NaN')
            #else:
            #    print('LL contribution:')
            #    print(self.LL[i])
            #    print('Data:')
            #    print(self.pair.data[i])
            #    print('Volatility:')
            #    print(self.act_h[i])
            #    print('Error:')
            #    print(self.error[i])
            #    print('//////////////////////////')

        print('attempt: ' + str(self.attempt))
        print('LLs:')
        print(self.LL)
        print('errors:')
        print(self.error)
        print('data:')
        print(self.pair.data)
        print('volatility:')
        print(self.act_h)

        params = [self.alpha, self.beta, self.gamma, self.delta, self.omega]

        Total_LL = numpy.nansum(self.LL)

        factor = 1.0
        crit = self.beta + self.alpha * self.gamma ** 2
        if crit > 1:
            factor = numpy.exp((crit - 1) * 10) * numpy.sign(Total_LL)
        elif crit < 0:
            factor = numpy.exp(-10 * crit) * numpy.sign(Total_LL)
        print('params:')
        print(params)

        print('crit:')
        print(crit)

        weighted_LL = factor * Total_LL
        if weighted_LL > 1000000:
            weighted_LL = 1000000

        print('LL sum')
        print(weighted_LL)
        print('////////////')
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

        self.attempt = 0

        #self.LL = numpy.zeros(len(self.pair.data))
        self.act_h = numpy.zeros(len(self.pair.data))
        self.error = numpy.zeros(len(self.pair.data))
        self.sim_h = numpy.zeros(self.steps)
        self.sim_R = numpy.zeros(self.steps)

        self.simulated_h()
        self.simulated_R()
        self.actual_h()
        #self.LogL(params)
