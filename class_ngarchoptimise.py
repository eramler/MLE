import iminuit
import class_ngarch

class Optimise(class_ngarch.NGARCH):

    def minimise(self):
        #defines function to be minimised, and parameters to be varied
        m = iminuit.Minuit(self.LogL, 
                                alpha = self.params[0],
                                error_alpha = self.params[0] / 10,
                                limit_alpha = (0, 1),
                                fix_alpha = False,

                                beta = self.params[1],
                                error_beta = self.params[1] / 10,
                                limit_beta = (0, 1),
                                fix_beta = False,

                                gamma = self.params[2],
                                error_gamma = self.params[2] / 10,
                                limit_gamma = (-500, 500),
                                fix_gamma = False,

                                delta = self.params[3],
                                error_delta = self.params[3] / 10,
                                limit_delta = (-100, 100),
                                fix_delta = False,

                                omega = self.params[4],
                                error_omega = self.params[4] / 10,
                                limit_omega = (0, 1),
                                fix_omega = False,
                                        
                                throw_nan = True,
                                pedantic = True,
                                print_level = 1
                                        )
        #runs minimiser and prints the optimised values
        m.migrad()
        print(m.values)
        print(m.errors)
        print(m.fval)
        #reruns minimisation from the fitted parameters until lowest LL value found
        if m.fval < self.lowest_LL:
            self.lowest_LL = m.fval
            self.fitted_params = [m.values[0], m.values[1], m.values[2], m.values[3], m.values[4]]
            self.params = self.fitted_params
            self.minimise()

        #m.draw_contour('alpha', 'gamma')

    #Class initialisation
    def __init__(self, pair, params, read_start=0, read_steps='MAX', forecast_start=0, forecast_steps='MAX'):
        class_ngarch.NGARCH.__init__(self, pair, params, read_start, read_steps, forecast_start, forecast_steps)
        self.params = params
        self.lowest_LL = 10000000 
        self.minimise()