import hdf5storage
import matplotlib.pyplot as pyplot
import math
import os

class CurrencyPair:

    #Function for cleaning the data, checking for NaN and infinities, sets to zero
    def dataclean(self):
        for r in range(0, len(self.data)-1):
            if math.isnan(self.data[r]) == True or math.isinf(self.data[r]) == True:
                self.data[r] = 0.0

    #Function to import .mat file, from relative /data/ folder, then calls cleaning function
    def importdata(self):
        directory = os.path.dirname(__file__)
        filename = os.path.join(directory, 'data/'+self.name+'.mat')

        self.data = hdf5storage.loadmat(filename)['ret']
        self.dataclean()

    #Function for displaying the data
    def figplot(self):
        pyplot.figure() 

        pyplot.title(self.name)
        pyplot.plot(self.data)

        pyplot.show()

    #Class initialisation function, calls functions within the class
    def __init__(self, name):
        self.name = name
        self.importdata()
        self.figplot()
