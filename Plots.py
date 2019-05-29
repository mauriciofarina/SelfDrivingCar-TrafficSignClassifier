import matplotlib.pyplot as plt
import numpy as np

def linePlot(x , y , xLabel = '' , yLabel = '' , fileName = 'test', save = False, show = False):
    
    plt.plot(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    if(save):
        plt.savefig('plots/' + fileName + '.png')
    
    if(show):
        plt.show()

def barPlot(x , y , xLabel = '' , yLabel = '' , fileName = 'test', stats = False, save = False, show = False):
    
    
    xLen = np.arange(len(x))
    plt.bar(xLen, y, align='center')
    plt.xticks(xLen, x)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    print("Test")
    if(stats):
        mean = np.mean(y)
        std = np.std(y)
        plt.title('$\mu=' + str(mean) + '$, $\sigma=' + str(std) + '$')
    print("Test")
    if(save):
        plt.savefig('plots/' + fileName + '.png')
    print("Test")
    if(show):
        plt.show()

def histogramPlot(x , bins = None , density = False , xLabel = '' , yLabel = '' , fileName = 'test', save = False, show = False, stats = True):
    
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




def plotTerminal(x, plot_height=10):
    ''' takes a list of ints or floats x and makes a simple terminal histogram.
        This function will make an inaccurate plot if the length of data list is larger than the number of columns
        in the terminal.'''

    result = []

    max_pos = 100

    x = np.array(x)

    x = (x*100)
    
    for i in x:
        temp = i/float(max_pos)
        temp2 = temp*plot_height
        result.append(int(temp2))



    hist_array = []

    col_list = []

    '''
    start = int((max_pos/plot_height))
    end = max_pos+1
    step = int((max_pos/plot_height))
    

    yAxis = range(start, end, step)
    for i in yAxis:
        temp = str(i).zfill(3)
        col_list.append(temp)

    hist_array.append(col_list)
    
    col_list = []
    yAxis = range(0, max_pos, int(max_pos/plot_height))
    for i in yAxis:
        col_list.append('     ')

    hist_array.append(col_list)

    '''

    maxVal = max(x)
    minVal = min(x)

    for idx, i in enumerate(result):
        col_list = []
        flag = True
        for j in range(plot_height):
            if j >= (i):
                if flag:
                    col_list.append("{:.1f} ".format(x[idx]))
                    flag = False
                else:
                    col_list.append('     ')
            else:
                if maxVal == x[idx]:
                    col_list.append("\u15CB\u15CB\u15CB\u15CB ")
                elif minVal == x[idx]:
                    col_list.append("\u15CA\u15CA\u15CA\u15CA ")
                else:
                    col_list.append("\u25A1\u25A1\u25A1\u25A1 ")
        

        hist_array.append(col_list)


    for i in reversed(range(len(hist_array[0]))):
        for j in range(len(hist_array)):
            print(hist_array[j][i], end='')
        print('')