from simul_large import Simul
from animatesimul_large import AnimateSimul
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(10)  # set random numbers to be always the same
    simulation = Simul(simul_time=0.2, sigma=1, L=60)  #  sigma particle radius # L box size
    print(simulation.__doc__)  # print the documentation from the class

    animate = AnimateSimul(simulation)
    animate.go(nframes=1000)  # number of animation steps
    print(simulation)  #  print last configuration to screen

if __name__ == '__main__':
    main()