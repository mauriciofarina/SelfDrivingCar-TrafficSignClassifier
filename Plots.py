import matplotlib.pyplot as plt
import numpy as np

def linePlot(x , y , xLabel = '' , yLabel = '' , fileName = 'test', save = False, show = True):
    
    plt.plot(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    if(save):
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        plt.show()

def barPlot(x , y , xLabel = '' , yLabel = '' , fileName = 'test', stats = False, save = False, show = True):
    
    xLen = np.arange(len(x))
    plt.bar(xLen, y, align='center')
    plt.xticks(xLen, x)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    if(stats):
        mean = np.mean(y)
        std = np.std(y)
        plt.title('$\mu=' + str(mean) + '$, $\sigma=' + str(std) + '$')

    if(save):
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        plt.show()

def histogramPlot(x , bins = None , density = False , xLabel = '' , yLabel = '' , fileName = 'test', save = False, show = True, stats = True):
    
    plt.hist(x, bins, align = 'mid', normed = density)
    
    probabilities = np.histogram(x, bins, density = density)
    
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    mean = np.mean(probabilities[0])
    std = np.std(probabilities[0])

    if(stats):
        plt.title('$\mu=' + str(mean) + '$, $\sigma=' + str(std) + '$')
    
    if(save):
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        plt.show()

    return mean, std