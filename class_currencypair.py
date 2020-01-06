import hdf5storage
import math
import os

class CurrencyPair:

    #Function to import .mat file, from relative /data/ folder
    def importdata(self):
        directory = os.path.dirname(__file__)
        filename = os.path.join(directory, 'data/'+self.name+'.mat')

        self.data = hdf5storage.loadmat(filename)['ret']

    #Function for cleaning the data, checking for NaN and infinities, sets to zero
    def dataclean(self):
        for i in range(0, len(self.data)):
            if math.isnan(self.data[i]) == True or math.isinf(self.data[i]) == True:
                self.data[i] = 0.0

    #Class initialisation
    def __init__(self, name):
        self.name = name
        self.importdata()
        self.dataclean()