import numpy

class VaryParams():
    #function to normally vary the parameters around their optimised values using the error as the standard deviation
    def sample_params(self):
        self.sampled_params = numpy.zeros((self.n_replicas, len(self.params)))
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
                self.sampled_params[i,4] = 0
            elif self.sampled_params[i,4] > 1:
                self.sampled_params[i,4] = 1
            

        #print(self.sampled_params)

    def __init__(self, params, param_errors, n_replicas):
        self.params = params
        self.param_errors = param_errors
        self.n_replicas = n_replicas
        self.sample_params()