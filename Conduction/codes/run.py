from simul import Simul
from animatesimul import AnimateSimul
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(10)  # set random numbers to be always the same
    simulation = Simul(simul_time=0.2, sigma=1, L=60)  #  sigma particle radius # L box size
    print(simulation.__doc__)  # print the documentation from the class

    animate = AnimateSimul(simulation)
    animate.go(nframes=1000)  # number of animation steps
    print(simulation)  #  print last configuration to screen
    
    plt.figure()
    plt.plot(simulation.graph_E)
    plt.show()

    P = np.polyfit(np.linspace(0,len(simulation.graph_E),len(simulation.graph_E)), simulation.graph_E, 1)
    print("polyfit = ", P)

if __name__ == '__main__':
    main()