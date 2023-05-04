from dataclasses import dataclass


@dataclass
class Controller:
    target_freq: float
    p_weight: float

    def get_ajustment(self, current_freq: float) -> float:
        return (self.target_freq - current_freq) * self.p_weight
