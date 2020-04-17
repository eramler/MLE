import hdf5storage
import math
import os
import numpy
import datetime

class CurrencyPair:
    #Function to import .mat file, from relative /data/ folder
    def importdata(self):
        directory = os.path.dirname(__file__)
        filename = os.path.join(directory, 'data/'+self.name+'.mat')
        self.daystamps = hdf5storage.loadmat(filename)['dv']
        self.daystamps = self.daystamps.flatten()
        self.data = hdf5storage.loadmat(filename)['ret']
        self.data = self.data.flatten()

    #Function for cleaning the data, checking for NaN and infinities, sets to zero
    def dataclean(self):
        for i in range(0, len(self.data)):
            if math.isnan(self.data[i]) == True or math.isinf(self.data[i]) == True:
                self.data[i] = 0.0

    #Function to convert the daynumbers in the database to timestamps
    def convertdates(self):
        self.dates = numpy.empty(len(self.daystamps), dtype='datetime64[s]')
        reference = datetime.datetime(1970, 1, 1)
        reference_daystamp = 719529  # days from 1-1-0000 to 1-1-1970
        for i in range(0, len(self.dates)):
            days_from_reference = self.daystamps[i] - reference_daystamp
            self.dates[i] = reference + datetime.timedelta(days=days_from_reference)

    #Class initialisation
    def __init__(self, name):
        self.name = name
        self.importdata()
        self.dataclean()
        self.convertdates()

CurrencyPair('EURGBP')