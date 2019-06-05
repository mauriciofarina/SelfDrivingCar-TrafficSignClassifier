'''
This file implements functions for ploting data.
'''

import matplotlib.pyplot as plt
import numpy as np


def linePlot(x , y , xLabel = '' , yLabel = '',setYAxis = None, grid = True,  fileName = 'test', save = False, show = False):
    
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    if (grid == True):
        plt.grid(True)
    
    if(setYAxis != None):
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,setYAxis[0],setYAxis[1]))

    if(save):
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        plt.show()

def barPlot(x , y , xLabel = '' , yLabel = '' , fileName = 'test', stats = False, save = False, show = False):
    
    plt.figure()
    xLen = np.arange(len(x))
    plt.bar(xLen, y, align='center', width=0.5)
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


def barPlot2(x , y , xLabel = '' , yLabel = '' ,setXAxis = None, setYAxis = None ,title = None,  fileName = 'test', stats = False, save = False, show = False):
    
    plt.figure(figsize=(12, 4))
    xLen = np.arange(len(x))
    plt.bar(xLen, y, align='center', width=0.5)
    plt.xticks(xLen, x)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(True)

    if(setXAxis != None):
        x1,x2,y1,y2 = plt.axis()
        plt.axis((setXAxis[0],setXAxis[1],y1,y2))
    
    if(setYAxis != None):
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,setYAxis[0],setYAxis[1]))
    
    if(title):
        plt.title(str(title))

    if(stats):
        mean = np.mean(y)
        std = np.std(y)
        plt.title('$\mu=' + str(mean) + '$, $\sigma=' + str(std) + '$')

    if(save):
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        plt.show()

def histogramPlot(x , bins = None , density = False , xLabel = '' , yLabel = '' , fileName = 'test', save = False, show = False, stats = True):
    
    plt.figure()
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



