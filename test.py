import numpy as np
import random
from matplotlib import pyplot as plt

def generate_seasonality(length: int = 100):
    max_period = random.randint(10, min(length // 2, 50))
    base_freq = 1 / max_period
    num_freqs = random.randint(1, 5)
    # possible_freq_multiples = [2**i for i in range(5)]
    possible_freq_multiples = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    freq_multiples = [1]
    additional_freqs = random.sample(possible_freq_multiples, num_freqs - 1)
    freq_multiples.extend(additional_freqs)
    freq_multiples.sort()

    trend_std = 1
    seasonality = np.zeros(length)
    phases = []
    amplitudes = []
    for freq_multiple in freq_multiples:
        phase = random.uniform(0, 2 * np.pi)
        phases.append(phase)
        amplitude = random.uniform(0.2, 0.4) * trend_std / (freq_multiple**0.5)
        amplitudes.append(amplitude)
        seasonality += amplitude * np.sin(
            2 * np.pi * freq_multiple * base_freq * np.arange(length) + phase
        )
    plt.plot(seasonality)
    plt.savefig("test.png")
    return max_period, num_freqs, additional_freqs, phases, amplitudes


if __name__ == "__main__":
    max_period, num_freqs, additional_freqs, phases, amplitudes = generate_seasonality()
    print(max_period, num_freqs, additional_freqs, phases, amplitudes)
