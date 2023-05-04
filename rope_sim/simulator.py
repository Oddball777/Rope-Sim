from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy import ndarray
from scipy.integrate import odeint

from .controller import Controller
from .rope import Rope


@dataclass
class Simulator:
    rope: Rope
    init_pos: ndarray
    init_vel: ndarray
    force_amp: ndarray
    mode: int
    controller: Optional[Controller] = None
    frequences: list[float] = field(default_factory=list)

    def get_simulation_data(self, time_start, time_end, number_of_steps):
        y0 = np.array(np.concatenate((self.init_pos, self.init_vel)))
        t = np.linspace(time_start, time_end, number_of_steps)
        wsol = odeint(self.differentiate_rope, y0, t)
        positions = wsol[:, : self.rope.number_of_masses]
        return t, positions

    def differentiate_rope(self, field, t):
        positions, velocities = (
            field[: self.rope.number_of_masses],
            field[self.rope.number_of_masses :],
        )
        dpos_dt = velocities
        dveloc_dt = (
            self.rope.get_omega_0**2
            * (
                -2 * positions
                + np.pad(positions, (1, 0))[:-1]
                + np.pad(positions, (0, 1))[1:]
            )
            - (self.rope.gamma * velocities)
            + (
                self.force_amp
                * np.sin(t * self.rope.get_omega_n(self.mode))
                / self.rope.mass_of_masses
            )
        )
        return np.concatenate((dpos_dt, dveloc_dt))

    def get_simulation_data_boucle(self, time_start, time_end, number_of_steps):
        y0 = np.array(np.concatenate((self.init_pos, self.init_vel)))
        t = np.linspace(time_start, time_end, number_of_steps)
        wsol = odeint(self.differentiate_rope_boucle, y0, t, mxstep=50000)
        positions = wsol[:, : self.rope.number_of_masses]
        return t, positions, np.array(self.frequences)

    def differentiate_rope_boucle(self, field, t):
        cur_freq = self.rope.get_f_n(1)
        self.frequences.append(cur_freq)
        if self.controller:
            self.rope.tension += self.controller.get_ajustment(cur_freq)
        positions, velocities = (
            field[: self.rope.number_of_masses],
            field[self.rope.number_of_masses :],
        )
        dpos_dt = velocities
        dveloc_dt = (
            self.rope.get_omega_0**2
            * (
                -2 * positions
                + np.pad(positions, (1, 0))[:-1]
                + np.pad(positions, (0, 1))[1:]
            )
            - (self.rope.gamma * velocities)
            + (
                self.force_amp
                * np.sin(t * self.rope.get_omega_n(self.mode))
                / self.rope.mass_of_masses
            )
        )
        return np.concatenate((dpos_dt, dveloc_dt))
