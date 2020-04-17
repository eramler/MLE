import numpy
import class_chisquared

class VaryParams():
    #function to normally vary the NGARCH parameters around their optimised values using the error as the standard deviation
    def sample_params(self):
        self.sampled_params = numpy.zeros((self.n_replicas, 5))
        for i in range(0, self.n_replicas):
            numpy.random.seed(2*i+1)
            self.sampled_params[i,:] = numpy.random.normal(self.params, self.param_errors)
            #enforce parameter bounds
            if self.sampled_params[i,0] < 0:
                self.sampled_params[i,0] = 0
            elif self.sampled_params[i,0] > 1:
                self.sampled_params[i,0] = 1
            if self.sampled_params[i,1] < 0:
                self.sampled_params[i,1] = 0
            elif self.sampled_params[i,1] > 1:
                self.sampled_params[i,1] = 1
            if self.sampled_params[i,2] < -500:
                self.sampled_params[i,2] = -500
            elif self.sampled_params[i,2] > 500:
                self.sampled_params[i,2] = 500
            #if self.sampled_params[i,3] < -100:
            #    self.sampled_params[i,3] = -100
            #elif self.sampled_params[i,3] > 100:
            #    self.sampled_params[i,3] = 100
            if self.sampled_params[i,4] < 0:
                self.sampled_params[i,4] = 1e-8
            elif self.sampled_params[i,4] > 1:
                self.sampled_params[i,4] = 1
            
    def run_replicas(self):
        self.replica_means = numpy.zeros((self.n_replicas, self.run_length))
        self.replica_stdevs = numpy.zeros((self.n_replicas, self.run_length))
        for i in range(0, self.n_replicas):
            chi = class_chisquared.ChiSquared(self.pair, self.sampled_params[i], self.run_start, self.run_length, n_runs=1000 ,showplot=False, n_histograms=0)
            for j in range(0, self.run_length):
                self.replica_means[i,j] = chi.mean_values[j]
                self.replica_stdevs[i,j] = chi.stdev_values[j]

    def calc_means(self):
        self.means = numpy.zeros(self.run_length)
        self.stdevs = numpy.zeros(self.run_length)
        for j in range(0, self.run_length):
            self.means[j] = numpy.sum(self.replica_means[:,j]) / self.n_replicas
            self.stdevs[j] = numpy.sum(self.replica_stdevs[:,j]) / self.n_replicas

    def reduced_chisquared(self):
        self.red_chi_array = numpy.zeros(self.run_length)
        for i in range(0, self.run_length):
            self.red_chi_array[i] = ((self.pair.data[self.run_start + i] - self.means[i]) / self.stdevs[i])** 2 / self.run_length
        self.red_chisquared = numpy.nansum(self.red_chi_array)

    def __init__(self, pair, params, param_errors, n_replicas=100, run_start=0, run_length='MAX'):
        self.pair = pair
        self.params = params
        self.param_errors = param_errors
        self.n_replicas = n_replicas
        self.run_start = run_start
        if run_length=='MAX':
            self.run_length = len(self.pair.data) - self.run_start
        else:
            self.run_length = run_length

        self.sample_params()
        self.run_replicas()
        self.calc_means()
        self.reduced_chisquared()