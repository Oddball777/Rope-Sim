from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import ndarray


@dataclass
class Animator:
    positions: ndarray
    times: ndarray
    frequences: ndarray

    def __post_init__(self):
        self.fig, self.ax = plt.subplots()
        self.num_masses = len(self.positions[0])
        self.ax.set_xlim(0, self.num_masses + 1)
        self.ax.set_ylim(
            1.2 * min(self.positions.flatten()), 1.2 * max(self.positions.flatten())
        )
        self.line = self.ax.plot([], [], "-", markersize=10)[0]
        self.text = self.ax.text(4, max(self.positions.flatten()), s="")

    def update(self, frame):
        # Set the positions of each mass to the values in the corresponding row of the positions array
        self.line.set_data(
            [i + 1 for i in range(self.num_masses)], self.positions[frame]
        )
        self.line.set_label(f"Frame = {frame}")
        self.text.set_text(s=f"Freq : {self.frequences[frame]}")
        plt.legend()
        return self.line

    def animate(self):
        animation = FuncAnimation(
            self.fig, self.update, frames=len(self.times), interval=1
        )
        plt.show()
