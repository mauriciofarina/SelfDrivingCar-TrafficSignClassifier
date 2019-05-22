import termplot
import numpy as np
import Plots as plot



terminalPlot = [0.5,0.6,0.7,0.4,0.5]

terminalPlot = np.array(terminalPlot)

terminalPlot = (terminalPlot*100)




plot.plotTerminal(terminalPlot, plot_height=20)