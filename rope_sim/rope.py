from dataclasses import dataclass

import numpy as np


@dataclass
class Rope:
    tension: float
    density: float
    gamma: float
    length: float
    number_of_masses: int

    def __post_init__(self):
        self.length_between_masses = self.length / self.number_of_masses
        self.mass_of_masses = self.density * self.length / self.number_of_masses

    @property
    def get_omega_0(self):
        return np.sqrt(
            self.tension / (self.length_between_masses * self.mass_of_masses)
        )

    def get_omega_n(self, mode_n: int):
        return (
            2
            * self.get_omega_0
            * np.sin(mode_n * np.pi / (2 * (self.number_of_masses + 1)))
        )

    def get_f_n(self, mode_n: int):
        return self.get_omega_n(mode_n) / (2 * np.pi)
