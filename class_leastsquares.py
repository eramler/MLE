import numpy
import class_ngarch
import class_currencypair
import matplotlib.pyplot as pyplot
import iminuit

class LeastSquares(class_ngarch.NGARCH):
    def rand_params(self):
        #set limits for the parameters to be varied within
        self.alphamin, self.alphamax = 0, 1e-4
        self.betamin, self.betamax = 0, 1
        self.gammamin, self.gammamax = -500, 500
        self.deltamin, self.deltamax = -10, 10
        self.omegamin, self.omegamax = 0, 1e-4
        #randomly select values for each parameter within their ranges, seeds for repeatability
        for i in range(0, self.n_randparams):
            numpy.random.seed(i)
            self.randparams[i,0] = numpy.random.uniform(self.alphamin, self.alphamax)
            numpy.random.seed(100000+i)
            self.randparams[i,1] = numpy.random.uniform(self.betamin,self.betamax)
            numpy.random.seed(200000+i)
            self.randparams[i,2] = numpy.random.uniform(self.gammamin,self.gammamax)
            numpy.random.seed(300000+i)
            self.randparams[i,3] = numpy.random.uniform(self.deltamin,self.deltamax)
            numpy.random.seed(400000+i)
            self.randparams[i,4] = numpy.random.uniform(self.omegamin,self.omegamax)


    def run_simulations(self):
        #create an array to store the values of the mean R values for each set of random parameters
        self.R_means = numpy.zeros((self.n_randparams, self.run_steps))
        self.R_stdevs = numpy.zeros((self.n_randparams, self.run_steps))
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
            #calculate the mean and standard deviation for R values for this iteration of random parameters
            for k in range(0, self.run_steps):
                self.R_means[i,k] = numpy.sum(R_simvalues[:,k]) / self.n_simulations
                self.R_stdevs[i,k] = numpy.sqrt(numpy.sum((R_simvalues[:,k] - self.R_means[i,k])**2) / self.n_simulations)
        #print(self.R_means)
        #print(self.R_stdevs)

    def squared_residuals(self):
        self.sqres_means = numpy.zeros((self.n_randparams, self.run_steps))
        self.sqres_Rfast = numpy.zeros((self.n_randparams, self.run_steps))
        for j in range(0, self.run_steps):
            self.sqres_means[:,j] = (self.R_means[:,j] - self.pair.data[j])**2
            self.sqres_Rfast[:,j] = (self.Rfast_values[:,j] - self.pair.data[j])**2
            for i in range(0, self.n_randparams): 
                if self.sqres_means[i,j] > 1000:
                    self.sqres_means[i,j] = 1000
                if self.sqres_Rfast[i,j] > 1000:
                    self.sqres_Rfast[i,j] = 1000
        self.Rfast_sqres_min_index = numpy.argmin(self.sqres_Rfast[:,3])
        #print(self.sqresiduals)

    def optimise_Rfast(self):
        #for clarity
        self.alphas = self.randparams[:,0]
        self.betas = self.randparams[:,1]
        self.gammas = self.randparams[:,2]
        self.deltas = self.randparams[:,3]
        self.omegas = self.randparams[:,4]
        self.Rfast_values = numpy.zeros((self.n_randparams, self.run_steps))

        bin_params = numpy.zeros(21)
        bin_param_step = numpy.ones(21)
        bin_param_names = ['v_c','v_A','v_B','v_G','v_D','v_O','v_AA','v_AB','v_AG','v_AD','v_AO','v_BB','v_BG','v_BD','v_BO','v_GG','v_GD','v_GO','v_DD','v_DO','v_OO']
        self.bin_i = 3
        #self.calc_Rfast(3, bin_params)
        #m = iminuit.Minuit(self.calc_Rfast, forced_parameters=bin_param_names, use_array_call=True)
        m = iminuit.Minuit.from_array_func(self.calc_Rfast, bin_params, error=bin_param_step, name=bin_param_names)
        for i in range(0, self.run_steps):
            self.bin_i = i
            m.migrad()
            print(m.fval)
            print(m.values)
        print(self.Rfast_values)

    def calc_Rfast(self, bin_params):
        #Setting up the parameters to be optimised in the 'fast' simulation
        #A: Alpha, B: Beta, G: Gamma, D: Delta, O: Omega, c: constant
        v_c,v_A,v_B,v_G,v_D,v_O,v_AA,v_AB,v_AG,v_AD,v_AO,v_BB,v_BG,v_BD,v_BO,v_GG,v_GD,v_GO,v_DD,v_DO,v_OO = bin_params

        #calculate Rfast values (multiline sum)
        self.Rfast_values[:,self.bin_i] = v_c + \
                            self.alphas * (v_A + v_AA*self.alphas + v_AB*self.betas + v_AG*self.gammas + v_AD*self.deltas + v_AO*self.omegas) + \
                            self.betas * (v_B + v_BB*self.betas + v_BG*self.gammas + v_BD*self.deltas + v_BO*self.omegas) + \
                            self.gammas * (v_G + v_GG*self.gammas + v_GD*self.deltas + v_GO*self.omegas) + \
                            self.deltas * (v_D + v_DD*self.deltas + v_DO*self.omegas) + \
                            self.omegas * (v_O + v_OO*self.omegas)
        
        Rfast_chivalues = numpy.zeros(self.n_randparams)
        for k in range(0, self.n_randparams):
            Rfast_chivalues[k] = ((self.Rfast_values[k,self.bin_i] - self.R_means[k,self.bin_i]) / self.R_stdevs[k,self.bin_i]) ** 2 / self.n_randparams
        #print(Rfast_chivalues)
        Rfast_totalchi = numpy.sum(Rfast_chivalues)
        #print(Rfast_totalchi)
        return Rfast_totalchi


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

        self.optimise_Rfast()

        self.squared_residuals()


pair = class_currencypair.CurrencyPair('EURGBP')
x = LeastSquares(pair, n_randparams=1000, n_simulations=100, run_steps=5)

pyplot.figure()
pyplot.plot(x.randparams[:,0], x.Rfast_values[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,0], x.R_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,0], x.Rfast_values[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('alpha value')
pyplot.ylabel('R Value')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,1], x.Rfast_values[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,1], x.R_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,1], x.Rfast_values[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('beta value')
pyplot.ylabel('R Value')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,2], x.Rfast_values[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,2], x.R_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,2], x.Rfast_values[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('gamma value')
pyplot.ylabel('R Value')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,3], x.Rfast_values[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,3], x.R_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,3], x.Rfast_values[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('delta value')
pyplot.ylabel('R Value')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,4], x.Rfast_values[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,4], x.R_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,4], x.Rfast_values[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('omega value')
pyplot.ylabel('R Value')
pyplot.show()

##################################PLOTTING SQUARED RESIDUALS###################################

pyplot.figure()
pyplot.plot(x.randparams[:,0], x.sqres_Rfast[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,0], x.sqres_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,0], x.sqres_Rfast[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('alpha value')
pyplot.ylabel(r'$(R_{sim} - R_{data})^2$')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,1], x.sqres_Rfast[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,1], x.sqres_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,1], x.sqres_Rfast[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('beta value')
pyplot.ylabel(r'$(R_{sim} - R_{data})^2$')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,2], x.sqres_Rfast[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,2], x.sqres_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,2], x.sqres_Rfast[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('gamma value')
pyplot.ylabel(r'$(R_{sim} - R_{data})^2$')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,3], x.sqres_Rfast[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,3], x.sqres_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,3], x.sqres_Rfast[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('delta value')
pyplot.ylabel(r'$(R_{sim} - R_{data})^2$')
#pyplot.show()

pyplot.figure()
pyplot.plot(x.randparams[:,4], x.sqres_Rfast[:,3], label='Rfast values', linestyle='', marker='o')
#pyplot.plot(x.randparams[:,4], x.sqres_means[:,3], label='Mean R values',linestyle='', marker='o')
pyplot.plot(x.randparams[x.Rfast_sqres_min_index,4], x.sqres_Rfast[x.Rfast_sqres_min_index,3], label='Minimum SqRes Value', linestyle='', marker='o')
pyplot.legend()
pyplot.xlabel('omega value')
pyplot.ylabel(r'(R_{sim} - R_{data})^2')
pyplot.show()


#for i in range(0, x.n_randparams):
#    pyplot.figure()
#    pyplot.plot(x.R_means[i,:])
#    pyplot.plot(x.R_means[i,:] + x.R_stdevs[i,:])
#    pyplot.plot(x.R_means[i,:] - x.R_stdevs[i,:])
#pyplot.show()