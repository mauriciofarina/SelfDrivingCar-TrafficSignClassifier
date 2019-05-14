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
    
    print('1')
    xLen = np.arange(len(x))
    print('1')
    plt.bar(xLen, y, align='center')
    print('1')
    plt.xticks(xLen, x)
    print('1')
    plt.xlabel(xLabel)
    print('1')
    plt.ylabel(yLabel)

    print('1')
    if(stats):
        print('2')
        mean = np.mean(y)
        print('2')
        std = np.std(y)
        print('2')
        plt.title('$\mu=' + str(mean) + '$, $\sigma=' + str(std) + '$')
        print('2')

    if(save):
        print('3')
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        print('4')
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