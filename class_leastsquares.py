import numpy
import class_ngarch
import class_currencypair
import matplotlib.pyplot as pyplot

class LeastSquares(class_ngarch.NGARCH):
    def rand_params(self):
        for i in range(0, self.n_randparams):
            numpy.random.seed(i)
            self.randparams[i,0] = numpy.random.uniform(0,1e-5)
            numpy.random.seed(100000+i)
            #numpy.random.seed(100000)
            self.randparams[i,1] = numpy.random.uniform(0,1)
            numpy.random.seed(200000+i)
            #numpy.random.seed(200000)
            self.randparams[i,2] = numpy.random.uniform(-500,500)
            numpy.random.seed(300000+i)
            #numpy.random.seed(300000)
            self.randparams[i,3] = numpy.random.uniform(10,-10)
            numpy.random.seed(400000+i)
            #numpy.random.seed(100000)
            self.randparams[i,4] = numpy.random.uniform(0,1e-5)
        #print(self.randparams[:,0])

    def run_simulations(self):
        #create an array to store the values of the mean R values for each set of random parameters
        self.R_means = numpy.zeros((self.n_randparams, self.run_steps))
        for i in range(0, self.n_randparams):
            print(i)
            #initialise the ngarch class with a set of random parameters
            class_ngarch.NGARCH.__init__(self, self.pair, self.randparams[i,:])
            #array to temporarily hold the R values for each simulation
            R_simvalues = numpy.zeros((self.n_simulations, self.run_steps))
            for j in range(0, self.n_simulations):
                self.simulated_h(rand_seed = j)
                self.simulated_R(rand_seed = 10000000 + j)
                R_simvalues[j,:] = self.sim_R[0:self.run_steps]
            #calculate the mean R values for this iteration of random parameters
            for k in range(0, self.run_steps):
                self.R_means[i,k] = numpy.sum(R_simvalues[:,k]) / self.n_simulations
        #print(self.R_means)

    def squared_residuals(self):
        self.sqresiduals = numpy.zeros((self.n_randparams, self.run_steps))
        for j in range(0, self.run_steps):
            self.sqresiduals[:,j] = (self.R_means[:,j] - self.pair.data[j])**2
            for i in range(0, self.n_randparams): 
                if self.sqresiduals[i,j] > 1000:
                    self.sqresiduals[i,j] = 1000
        print(self.sqresiduals)

    def __init__(self, pair, n_randparams=10, n_simulations=10, run_steps='MAX'):
        self.pair = pair
        self.n_randparams = n_randparams
        self.n_simulations = n_simulations
        if run_steps=='MAX':
            self.run_steps = len(self.pair.data)
        else:
            self.run_steps = run_steps

        self.randparams = numpy.zeros((self.n_randparams, 5))
        
        self.rand_params()
        self.run_simulations()
        self.squared_residuals()


pair = class_currencypair.CurrencyPair('EURGBP')
x = LeastSquares(pair, n_randparams=100, n_simulations=100, run_steps=5)

pyplot.figure()
pyplot.plot(x.randparams[:,0], x.sqresiduals[:,4], linestyle='', marker='o')
pyplot.xlabel('alpha value')
pyplot.ylabel('R value')
pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,1], x.sqresiduals[:,4], linestyle='', marker='o')
pyplot.xlabel('beta value')
pyplot.ylabel('R value')
pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,2], x.sqresiduals[:,4], linestyle='', marker='o')
pyplot.xlabel('gamma value')
pyplot.ylabel('R value')
pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,3], x.sqresiduals[:,4], linestyle='', marker='o')
pyplot.xlabel('delta value')
pyplot.ylabel('R value')
pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,4], x.sqresiduals[:,4], linestyle='', marker='o')
pyplot.xlabel('omega value')
pyplot.ylabel('R value')
pyplot.show()