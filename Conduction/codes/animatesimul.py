import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import EllipseCollection, LineCollection

from simul import Simul


class AnimateSimul:
    def __init__(self, simulation):
        self.simulation = simulation
        self.fig, self.ax = plt.subplots(figsize=(5, 5))  # initialise  graphics
        self.circles = EllipseCollection(widths=2*simulation.sigma, heights=2*simulation.sigma, angles=0, units='x',
                                         offsets=simulation.position, transOffset=self.ax.transData)  # circles at position
        self.ax.add_collection(self.circles)

        self.segment = [[[0, 0], [0, simulation.L], [simulation.L, simulation.L], [simulation.L, 0], [0, 0]]]  # simulation cell
        self.line = LineCollection(self.segment, colors='#000000')  # draw square
        self.ax.add_collection(self.line)

        self.ax.set_xlim(left=-0.5, right=self.simulation.L+0.5)  # plotting limits on screen
        self.ax.set_ylim(bottom=-0.5, top=self.simulation.L+0.5)
        self._ani = 0


    def init(self):  # this is the first thing drawn to the screen
        self.circles.set_offsets(self.simulation.position)


    def _anim_step(self, m):  # m is the number of calls that have occurred to this function
        print('anim_step m = ', m)
        if m == 0:
            time.sleep(1)

        self.simulation.md_step()  # perform simulation step
        self.circles.set_offsets(self.simulation.position)  # update positions on screen


    def go(self, nframes):
        self._ani = animation.FuncAnimation(self.fig, func=self._anim_step, frames=nframes,
                                            repeat=False, interval=10, init_func=self.init)  # run animation
        plt.show()