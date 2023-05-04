import numpy as np


def get_omega_0(tension, length_between_masses, mass_of_masses):
    return np.sqrt(tension / (length_between_masses * mass_of_masses))


def get_omega_n(mode_n, omega_0, number_of_masses):
    return 2 * omega_0 * np.sin(mode_n * np.pi / (2 * (number_of_masses + 1)))


def get_f_n(omega_n):
    return omega_n / (2 * np.pi)


TENSION = 70
LENGTH = 0.43
DENSITY = 0.0062
NUMBER_OF_MASSES = 15
NUMBER_OF_SEGMENTS = NUMBER_OF_MASSES + 1
LENGTH_BETWEEN_MASSES = LENGTH / NUMBER_OF_SEGMENTS
MASS_OF_MASSES = DENSITY * LENGTH / NUMBER_OF_MASSES
MODE = 1
omega_0 = get_omega_0(TENSION, LENGTH_BETWEEN_MASSES, MASS_OF_MASSES)
omega_n = get_omega_n(MODE, omega_0, NUMBER_OF_MASSES)
frequency_n = get_f_n(omega_n)
print(frequency_n)
print(LENGTH_BETWEEN_MASSES, MASS_OF_MASSES)
print(omega_0)
