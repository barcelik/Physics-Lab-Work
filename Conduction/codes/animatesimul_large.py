import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection, LineCollection
import numpy as np


from simul_large import Simul


class AnimateSimul:
    def __init__(self, simulation):
        self.simulation = simulation
        self.fig, self.ax = plt.subplots(figsize=(8, 8))  # initialise  graphics
        self.circles = EllipseCollection(widths=[2*self.simulation.sigma] * len(self.simulation.position) +
                                          [2*self.simulation.sigma2],
                                 heights=[2*self.simulation.sigma] * len(self.simulation.position) +
                                           [2*self.simulation.sigma2],
                                 angles=[0] * (len(self.simulation.position) + 1), units='x',
                                 offsets=np.vstack([self.simulation.position, self.simulation.position2]),
                                 transOffset=self.ax.transData)

        
        self.ax.add_collection(self.circles)

        self.segment = [[[0, 0], [0, simulation.L], [simulation.L, simulation.L], [simulation.L, 0], [0, 0]]]  # simulation cell
        self.line = LineCollection(self.segment, colors='#000000')  # draw square
        self.ax.add_collection(self.line)

        self.ax.set_xlim(left=-0.5, right=self.simulation.L+0.5)  # plotting limits on screen
        self.ax.set_ylim(bottom=-0.5, top=self.simulation.L+0.5)
        self._ani = 0


    def init(self):  # this is the first thing drawn to the screen
        self.circles.set_offsets(np.vstack([self.simulation.position, self.simulation.position2]))



    def _anim_step(self, m):  # m is the number of calls that have occurred to this function
        print('anim_step m = ', m)
        if m == 0:
            time.sleep(1)

        self.simulation.md_step()  # perform simulation step
        self.circles.set_offsets(np.vstack([self.simulation.position, self.simulation.position2]))
  # update positions on screen


    def go(self, nframes):
        self._ani = animation.FuncAnimation(self.fig, func=self._anim_step, frames=nframes,
                                            repeat=False, interval=10, init_func=self.init)  # run animation
        plt.show()