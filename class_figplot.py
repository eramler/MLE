import matplotlib.pyplot as pyplot

class Figplot:

    #Function for plotting the data
    def plot(self):
        pyplot.figure() 

        pyplot.title(self.name)
        pyplot.plot(self.data)

        pyplot.show()

    #Class initialisation
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.plot()