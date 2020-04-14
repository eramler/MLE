import class_ngarch
import numpy
import matplotlib.pyplot as pyplot
import scipy.stats
import class_figplot

class ChiSquared(class_ngarch.NGARCH):
    '''
run_start and run_steps should normally be same as the forecast steps in the NGARCH class.
    '''
    #simulates the volatility and then returns, which are stored for each datapoint across multiple runs
    def run_simulation(self):
        #2x2 array for each value of the forecast across the number of runs
        self.sim_values = numpy.zeros((self.run_steps, self.n_runs))
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
            self.stdev_values[i] = numpy.sqrt((numpy.sum((self.sim_values[i,:] - self.mean_values[i])**2 )) / self.n_runs)


    #shows the histogras for the binned returns values for each datapoint
    def show_histograms(self):
        if self.n_histograms > self.run_steps:
            self.n_histograms = self.run_steps
            print('n_histograms greater than n_steps, showing maximum number of histograms')
        
        for i in range(0, self.n_histograms):
            normal_range = numpy.linspace(self.mean_values[i] - 4*self.stdev_values[i], self.mean_values[i] + 4*self.stdev_values[i], 100)
            normal_values = scipy.stats.norm.pdf(normal_range, self.mean_values[i], self.stdev_values[i])

            n_bins = numpy.int(self.n_runs/100)
            if n_bins < 20:
                n_bins = 20

            n, bins, patches = pyplot.hist(self.sim_values[i,:], n_bins)

            bin_width = bins[1] - bins[0]
            scale_factor = self.n_runs * bin_width

            pyplot.plot(normal_range, normal_values * scale_factor)
            pyplot.title('Binning Simulated Values: '+str(self.n_runs)+' runs')
            pyplot.xlabel('Simulated Values')
            pyplot.ylabel('Bin Population')
            ax = pyplot.gca()
            ax.text(-0.018, 300, 'Mean: '+str(self.mean_values[i]))
            ax.text(-0.018, 280, 'Standard Deviation: '+str(self.stdev_values[i]))
            pyplot.show()

    def calc_reducedchi(self):
        self.chi_values = numpy.zeros(self.run_steps)
        self.chi_values = ((self.pair.data[self.run_start:self.run_start+self.run_steps] - self.mean_values) / self.stdev_values)**2 / self.read_steps
        self.reducedchi_value = numpy.sum(self.chi_values)
        print('Reduced Chi Value:', self.reducedchi_value)

    def __init__(self, pair, params, run_start, run_steps, n_runs=100, showplot=False, n_histograms=3):
        class_ngarch.NGARCH.__init__(self, pair, params, read_start=run_start, read_steps=run_steps, forecast_start=run_start, forecast_steps=run_steps)
        self.run_start = run_start
        self.run_steps = run_steps
        self.n_runs = n_runs
        self.n_histograms = n_histograms

        self.run_simulation()
        self.calc_mean_stdev()
        self.calc_reducedchi()

        if showplot == True:
            self.show_histograms()
            #class_figplot.Figplot(self.pair.name+' Chi: '+str(self.reducedchi_value), [self.pair.data[self.run_start:self.run_start+self.run_steps], self.mean_values], ['Real Data', 'Mean Values'], 'points', 'values')
            #class_figplot.Figplot(self.pair.name, [self.mean_values], ['Mean Values'], 'points', 'values')
            #class_figplot.Figplot(self.pair.name+' Chi: '+str(self.reducedchi_value), [self.stdev_values], ['Standard Deviation Values'], 'points', 'values')
            class_figplot.Figplot(self.pair.name+r' $\chi^2$: '+str(self.reducedchi_value), [self.pair.data[self.run_start:self.run_start+self.run_steps], self.mean_values, self.mean_values + self.stdev_values, self.mean_values - self.stdev_values], ['True Data', 'Simulated Mean', r'Mean + 1$\sigma$', r'Mean - 1$\sigma$'], 'Simulated Points', 'Returns')
            #pyplot.show()