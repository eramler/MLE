import numpy
import class_ngarch
import class_currencypair
import matplotlib.pyplot as pyplot
import iminuit

class LeastSquares(class_ngarch.NGARCH):
    def rand_params(self):
        self.randparams = numpy.zeros((self.n_randparams, 5))
        #set limits for the parameters to be varied within
        self.alphamin, self.alphamax = 0, 1e-5
        self.betamin, self.betamax = 0, 1
        self.gammamin, self.gammamax = -500, 500
        self.deltamin, self.deltamax = -10, 10
        self.omegamin, self.omegamax = 1e-8, 1e-4
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
        self.R_means = numpy.zeros((self.n_randparams, self.read_steps))
        self.R_stdevs = numpy.zeros((self.n_randparams, self.read_steps))
        for i in range(0, self.n_randparams):
            #print(i)
            #initialise the ngarch class with a set of random parameters
            class_ngarch.NGARCH.__init__(self, self.pair, self.randparams[i,:], self.read_start, self.read_steps, self.read_start, self.read_steps)
            #array to temporarily hold the R values for each simulation
            R_simvalues = numpy.zeros((self.n_simulations, self.read_steps))
            for j in range(0, self.n_simulations):
                self.simulated_h(rand_seed = j)
                self.simulated_R(rand_seed = 10000000 + j)
                R_simvalues[j,:] = self.sim_R
            #calculate the mean and standard deviation for R values for this iteration of random parameters
            for k in range(0, self.read_steps):
                self.R_means[i,k] = numpy.nansum(R_simvalues[:,k]) / self.n_simulations
                #self.R_means[i,k] = numpy.sum(R_simvalues[:,k]) / self.n_simulations
                self.R_stdevs[i,k] = numpy.sqrt(numpy.nansum((R_simvalues[:,k] - self.R_means[i,k])**2) / self.n_simulations)
                #self.R_stdevs[i,k] = numpy.sqrt(numpy.sum((R_simvalues[:,k] - self.R_means[i,k])**2) / self.n_simulations)
        #print(self.R_means)
        #print(self.R_stdevs)

    def optimise_Rfast(self):
        #for clarity
        self.alphas = self.randparams[:,0]
        self.betas = self.randparams[:,1]
        self.gammas = self.randparams[:,2]
        self.deltas = self.randparams[:,3]
        self.omegas = self.randparams[:,4]
        self.Rfast_values = numpy.zeros((self.n_randparams, self.read_steps))

        bin_params = numpy.zeros(21)
        bin_param_step = numpy.ones(21)
        self.bin_param_values = numpy.zeros((self.read_steps,21))
        self.bin_param_errors = numpy.zeros((self.read_steps, 21))
        self.bin_param_names = ['v_c','v_A','v_B','v_G','v_D','v_O','v_AA','v_AB','v_AG','v_AD','v_AO','v_BB','v_BG','v_BD','v_BO','v_GG','v_GD','v_GO','v_DD','v_DO','v_OO']
        self.bin_i = 0
        #m = iminuit.Minuit(self.calc_Rfast, forced_parameters=bin_param_names, use_array_call=True)
        m = iminuit.Minuit.from_array_func(self.calc_Rfast, bin_params, error=bin_param_step, name=self.bin_param_names, errordef=1)
        for i in range(0, self.read_steps):
            m.migrad()
            #print(m.fval)
            self.bin_param_values[i,:] = m.args
            #print(m.errors)
            for j in range(0, 21):
                self.bin_param_errors[i,j] = m.errors[j]
            self.bin_i = self.bin_i + 1
        #print(self.bin_param_values)
        #print(self.bin_param_errors)
        #print(self.Rfast_values)
        #print(self.bin_param_values)

        #self.e_c, self.e_A, self.e_B = self.bin_param_errors[:,0], self.bin_param_errors[:,1], self.bin_param_errors[:,2]
        #self.e_G, self.e_D, self.e_O = self.bin_param_errors[:,3], self.bin_param_errors[:,4], self.bin_param_errors[:,5]
        #self.e_AA, self.e_AB, self.e_AG = self.bin_param_errors[:,6], self.bin_param_errors[:,7], self.bin_param_errors[:,8]
        #self.e_AD, self.e_AO, self.e_BB = self.bin_param_errors[:,9], self.bin_param_errors[:,10], self.bin_param_errors[:,11]
        #self.e_BG, self.e_BD, self.e_BO = self.bin_param_errors[:,12], self.bin_param_errors[:,13], self.bin_param_errors[:,14]
        #self.e_GG, self.e_GD, self.e_GO = self.bin_param_errors[:,15], self.bin_param_errors[:,16], self.bin_param_errors[:,17]
        #self.e_DD, self.e_DO, self.e_OO = self.bin_param_errors[:,18], self.bin_param_errors[:,19], self.bin_param_errors[:,20]

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
            Rfast_chivalues[k] = ((self.Rfast_values[k,self.bin_i] - self.R_means[k,self.bin_i]) / self.R_stdevs[k,self.bin_i]) ** 2
        #print(Rfast_chivalues)
        #print(numpy.sum(Rfast_chivalues))
        Rfast_totalchi = numpy.nansum(Rfast_chivalues)
        return Rfast_totalchi

    def optimise_params(self):
        self.v_c, self.v_A, self.v_B = self.bin_param_values[:,0], self.bin_param_values[:,1], self.bin_param_values[:,2]
        self.v_G, self.v_D, self.v_O = self.bin_param_values[:,3], self.bin_param_values[:,4], self.bin_param_values[:,5]
        self.v_AA, self.v_AB, self.v_AG = self.bin_param_values[:,6], self.bin_param_values[:,7], self.bin_param_values[:,8]
        self.v_AD, self.v_AO, self.v_BB = self.bin_param_values[:,9], self.bin_param_values[:,10], self.bin_param_values[:,11]
        self.v_BG, self.v_BD, self.v_BO = self.bin_param_values[:,12], self.bin_param_values[:,13], self.bin_param_values[:,14]
        self.v_GG, self.v_GD, self.v_GO = self.bin_param_values[:,15], self.bin_param_values[:,16], self.bin_param_values[:,17]
        self.v_DD, self.v_DO, self.v_OO = self.bin_param_values[:,18], self.bin_param_values[:,19], self.bin_param_values[:,20]

        self.initial_params = (1e-6, 0.5, 50, 1e-5, 1e-6)
        m = iminuit.Minuit(self.calc_params,
                            alpha = self.initial_params[0],
                            error_alpha = self.initial_params[0] / 10,
                            limit_alpha = (0,1),
                            
                            beta = self.initial_params[1],
                            error_beta = self.initial_params[1] / 10,
                            limit_beta = (0,1),
                            
                            gamma = self.initial_params[2],
                            error_gamma = self.initial_params[2] / 10,
                            limit_gamma = (-500, 500),
                            
                            delta = self.initial_params[3],
                            error_delta = self.initial_params[3] / 10,
                            limit_delta = (-100, 100),
                            
                            omega = self.initial_params[4],
                            error_omega = self.initial_params[4] / 10,
                            limit_omega = (1e-8,1),

                            print_level = 0,
                            errordef=1
                                                )
        m.migrad()
        self.fitted_params = [m.values[0], m.values[1], m.values[2], m.values[3], m.values[4]]
        #self.fitted_errors = [m.errors[0], m.errors[1], m.errors[2], m.errors[3], m.errors[4]]
        #print(m.fval)
        #print(m.values)
        #print(m.errors)
        

    def calc_params(self, alpha, beta, gamma, delta, omega):
        Rfast_binvalues = self.v_c + \
                            alpha * (self.v_A + self.v_AA*alpha + self.v_AB*beta + self.v_AG*gamma + self.v_AD*delta + self.v_AO*omega) + \
                            beta * (self.v_B + self.v_BB*beta + self.v_BG*gamma + self.v_BD*delta + self.v_BO*omega) + \
                            gamma * (self.v_G + self.v_GG*gamma + self.v_GD*delta + self.v_GO*omega) + \
                            delta * (self.v_D + self.v_DD*delta + self.v_DO*omega) + \
                            omega * (self.v_O + self.v_OO*omega)

#        Rfast_binerrors = numpy.sqrt( (self.e_c)**2 + \
#                            (alpha**2) * ((self.e_A)**2 + (self.e_AA*alpha)**2 + (self.e_AB*beta)**2 + (self.e_AG*gamma)**2 + (self.e_AD*delta)**2 + (self.e_AO*omega)**2) + \
#                            (beta**2) * ((self.e_B)**2 + (self.e_BB*beta)**2 + (self.e_BG*gamma)**2 + (self.e_BD*delta)**2 + (self.e_BO*omega)**2) + \
#                            (gamma**2) * ((self.e_G)**2 + (self.e_GG*gamma)**2 + (self.e_GD*delta)**2 + (self.e_GO*omega)**2) + \
#                            (delta**2) * ((self.e_D)**2 + (self.e_DD*delta)**2 + (self.e_DO*omega)**2) + \
#                            (omega**2) * ((self.e_O)**2 + (self.e_OO*omega)**2)    )
        
        Rfast_chi_sum = numpy.nansum((self.pair.data[self.read_start:self.read_start+self.read_steps] - Rfast_binvalues)**2)
        
        #Penalty term to enforce parameter bounds
        penalty = 0.0
        crit = beta + alpha * gamma ** 2
        if crit > 1:
            penalty = numpy.exp((crit - 1) * 10000) - 1
        elif crit < 0:
            penalty = numpy.exp(-10000 * crit) - 1

        weighted_Rfast_chi_sum = Rfast_chi_sum + penalty
        if weighted_Rfast_chi_sum > 1000000:
            weighted_Rfast_chi_sum = 1000000

        #print(weighted_Rfast_chi_sum)
        return weighted_Rfast_chi_sum

    def calc_param_errors(self):
        n_replicas = 100
        bin_param_replicas = numpy.zeros((n_replicas, self.read_steps, 21))
        for j in range(0, self.read_steps):
            for k in range(0, 21):
                numpy.random.seed(1000*j + k)
                bin_param_replicas[:,j,k] = numpy.random.uniform(low=(self.bin_param_values[j,k]-self.bin_param_errors[j,k]), high=(self.bin_param_values[j,k]+self.bin_param_errors[j,k]), size=n_replicas)

        #for i in range(0, n_replicas):




    def __init__(self, pair, n_randparams=10, n_simulations=10, read_start=0, read_steps='MAX'):
        self.pair = pair
        self.n_randparams = n_randparams
        self.n_simulations = n_simulations
        self.read_start = read_start
        if read_steps=='MAX':
            self.read_steps = len(self.pair.data) - self.read_start
        else:
            self.read_steps = read_steps

        self.rand_params()
        self.run_simulations()
        self.optimise_Rfast()
        self.optimise_params()
        self.calc_param_errors()