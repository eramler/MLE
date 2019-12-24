import iminuit
import class_ngarch

class Optimise(class_ngarch.NGARCH):

    def minimise(self):
        m = iminuit.Minuit(self.LogL, 
                                alpha = self.params[0],
                                error_alpha = self.params[0] / 10,
                                limit_alpha = (0, 20),
                                fix_alpha = False,

                                beta = self.params[1],
                                error_beta = self.params[1] / 10,
                                limit_beta = (0, 10),
                                fix_beta = False,

                                gamma = self.params[2],
                                error_gamma = self.params[2] / 10,
                                limit_gamma = (-500, 500),
                                fix_gamma = False,

                                delta = self.params[3],
                                error_delta = self.params[3] / 10,
                                limit_delta = (0, 1),
                                fix_delta = False,

                                omega = self.params[4],
                                error_omega = self.params[4] / 10,
                                limit_omega = (0.00000001, 1),
                                fix_omega = False,
                                        
                                throw_nan = True,
                                pedantic = True
                                        )
        m.migrad()
        self.fitted_params = m.values
        print(m.values)
        


    def __init__(self, pair, params, steps):
        class_ngarch.NGARCH.__init__(self, pair, params, steps)
        self.params = params

        self.minimise()