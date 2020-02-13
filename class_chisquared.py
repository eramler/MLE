import class_ngarch
import numpy
import matplotlib.pyplot as pyplot
import scipy.stats

class ChiSquared(class_ngarch.NGARCH):
    '''
run_start and run_steps should normally be same as the forecast steps in the NGARCH class.
    '''
    #simulates the volatility and then returns, which are stored for each datapoint across multiple runs
    def run_simulation(self):
        #2x2 array for each value of the forecast across the number of runs
        self.sim_values = numpy.zeros((self.run_steps, self.n_runs))
        #setting unique seeds for each simulation (providing less than 10 million runs)
        for j in range(0, self.n_runs):
            self.simulated_h(rand_seed = j)
            self.simulated_R(rand_seed = 10000000 + j)
            for i in range(0, self.run_steps):
                self.sim_values[i,j] = self.sim_R[i]


    def calc_mean_stdev(self):
        #calculate the mean value from all runs
        self.mean_values = numpy.zeros(self.run_steps)
        self.stdev_values = numpy.zeros(self.run_steps)
        for i in range(0, self.run_steps):
            self.mean_values[i] = numpy.sum(self.sim_values[i,:]) / self.n_runs
            self.stdev_values[i] = numpy.sqrt((numpy.sum((self.pair.data[self.read_start+i] - self.mean_values[i])**2 )) / self.n_runs)


    #shows the histogras for the binned returns values for each datapoint (default set to 10 to avoid overload if many points forecasted)
    def show_histograms(self, n_histograms=3):
        for i in range(0, n_histograms):
            normal_range = numpy.linspace(self.mean_values[i] - 4*self.stdev_values[i], self.mean_values[i] + 4*self.stdev_values[i], 100)
            normal_values = scipy.stats.norm.pdf(normal_range, self.mean_values[i], self.stdev_values[i])
            pyplot.hist(self.sim_values[i], 10)
            pyplot.plot(normal_range, normal_values)
            pyplot.show()


    def __init__(self, pair, params, run_start, run_steps, n_runs=10 ,showplot=False):
        class_ngarch.NGARCH.__init__(self, pair, params, read_start=run_start, read_steps=run_steps, forecast_start=run_start, forecast_steps=run_steps)
        self.run_start = run_start
        self.run_steps = run_steps
        self.n_runs = n_runs

        self.run_simulation()
        self.calc_mean_stdev()

        if showplot == True:
            self.show_histograms()