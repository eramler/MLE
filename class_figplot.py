import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import matplotlib

class Figplot:
    """
To plot multiple plots on the same graph, place in a list eg. [plot1, plot2]
labels: corresponding legend labels
    """

    #Function for plotting the data
    def plot(self):
        pyplot.figure() 

        font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 16}
        matplotlib.rc('font', **font)

        pyplot.title(self.name)
        for i in range(0, len(self.data)):
            pyplot.plot(self.data[i])

        pyplot.legend(self.labels)
        pyplot.xlabel(self.xlabel)
        pyplot.ylabel(self.ylabel)

    #Class initialisation
    def __init__(self, name, data, labels, xlabel, ylabel):
        self.name = name
        self.data = data
        self.labels = labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot()

#Function to show plots all at once after the Figplot class has finished being called
def showplot():
    pyplot.show()