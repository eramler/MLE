import iminuit
import class_ngarch

class ParamOptimise(class_ngarch.NGARCH):
    def minimize(self):
        print(self.alpha)
        iminuit.minimize(self.loglikelihood(), [self.alpha, self.beta, self.gamma, self.delta, self.omega], args=(self.pair.data) )

    def __init__(self, pair, params, steps):
        class_ngarch.NGARCH.__init__(self, pair, params, steps)


        self.minimize()



    