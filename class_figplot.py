import matplotlib.pyplot as pyplot

class Figplot:
    """
To plot multiple plots on the same graph, place in a list eg. [plot1, plot2]
labels: corresponding legend labels
    """

    #Function for plotting the data
    def plot(self):
        pyplot.figure() 

        pyplot.title(self.name)
        for i in range(0, len(self.data)):
            pyplot.plot(self.data[i])

        pyplot.legend(self.labels)

    #Class initialisation
    def __init__(self, name, data, labels):
        self.name = name
        self.data = data
        self.labels = labels
        self.plot()

#Function to show plots all at once after the Figplot class has finished being called
def showplot():
    pyplot.show()