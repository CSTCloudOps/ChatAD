# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
from tqdm import tqdm
import re
import json
from typing import *
from trend_utils import generate_random_points, generate_trend_prompt, generate_trend_curve, generate_trend_list
import matplotlib.pyplot as plt

# Config
ENABLE_MULTIPLE_TREND = True
ENABLE_DROP_PROMPT = True
ENABLE_MULTIPLE_SEASONAL = False
ENABLE_MULTIPLE_NOISE = False

# All Config for TS Attributes
# Notes
# 1. Seaonal and Frequency can be combined (e.g., high frequency-sin periodic fluctuation, so there should be 7 types of seasonal attributes)
# 2. We implemented 2 types of noise types (e.g., random noise, sin noise), see the codes below for more information
# 3. The value following the overall attribute is the probability of the attribute to be selected
# All Attribute Set
all_attribute_set = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.6,
            "sin periodic fluctuation": 0.35,
            "square periodic fluctuation": 0.02,
            "triangle periodic fluctuation": 0.03
        },
        "trend": {
            "decrease": 0.2,
            "increase": 0.2,
            "keep steady": 0.3,
            "multiple": 0.3
        },
        "frequency": {
            "high frequency": 0.5,
            "low frequency": 0.5
        },
        "noise": {
            "noisy": 0.3,
            "almost no noise": 0.7
        }
    },
    "change": {
        "shake": 2,
        "upward spike": 12,
        "downward spike": 10,
        "continuous upward spike": 4,
        "continuous downward spike": 2,
        "upward convex": 2,
        "downward convex": 2,
        "sudden increase": 2,
        "sudden decrease": 2,
        "rapid rise followed by slow decline": 2,
        "slow rise followed by rapid decline": 2,
        "rapid decline followed by slow rise": 2,
        "slow decline followed by rapid rise": 2,
        "decrease after upward spike": 3,
        "increase after downward spike": 3,
        "increase after upward spike": 3,
        "decrease after downward spike": 3,
        "wide upward spike": 3,
        "wide downward spike": 3
    }
}

def generate_random_attributes(overall_attribute: Dict[str, Dict[str, float]], change_attribute: Dict[str, float], change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None, seq_len: int = 512):
    if change_positions is None:
        change_positions = [(None, None) for _ in range(random.randint(0, 3))]
    description = {}

    description["seasonal"] = {
        "type": np.random.choice(list(overall_attribute['seasonal']), p=np.array(list(overall_attribute['seasonal'].values()))/sum(list(overall_attribute['seasonal'].values())))
    }

    trend_candidates = overall_attribute['trend'].copy()
    if not ENABLE_MULTIPLE_TREND and "multiple" in trend_candidates:
        trend_candidates.pop("multiple")
    trend_char = np.random.choice(list(trend_candidates), p=np.array(list(trend_candidates.values()))/sum(list(trend_candidates.values())))
    
    description["trend"] = {
        "type": trend_char
    }

    num_local_chars = len(change_positions)
    local_chars = list(np.random.choice(list(change_attribute), size=num_local_chars, p=np.array(list(change_attribute.values()))/sum(list(change_attribute.values()))))
    
    description["local"] = []
    for char in local_chars:
        local_position, local_amplitude = change_positions.pop()
        description["local"].append({
            "type": char,
            "position_start": local_position,
            "amplitude": local_amplitude
        })

    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        description["frequency"] = {'type': np.random.choice(list(overall_attribute['frequency']), p=np.array(list(overall_attribute['frequency'].values()))/sum(list(overall_attribute['frequency'].values())))}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    description["noise"] = {'type': np.random.choice(list(overall_attribute['noise']), p=np.array(list(overall_attribute['noise'].values()))/sum(list(overall_attribute['noise'].values())))}

    return description

def generate_controlled_attributes(attribute_set, change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None):
    if change_positions is None:
        # change_positions = [(None, None) for _ in range(random.randint(0, 2))]
        change_positions = [(None, None) for _ in range(np.random.choice([0,1,2], p=[0.5, 0.45, 0.05]))]
    description = {}

    seasonal_p = [all_attribute_set['overall_attribute']['seasonal'][i] for i in attribute_set['seasonal']['attributes']]
    # print(seasonal_p, attribute_set['seasonal']['amplitude']['min'], attribute_set['seasonal']['amplitude']['max'])
    # description["seasonal"] = {
    #     "type": np.random.choice(list(attribute_set['seasonal']['attributes']), p=np.array(seasonal_p)/sum(seasonal_p)),
    #     "amplitude": random.uniform(attribute_set['seasonal']['amplitude']['min'], attribute_set['seasonal']['amplitude']['max'])
    # }
    if attribute_set['seasonal']['amplitude']['min'] != attribute_set['seasonal']['amplitude']['min']:
        description["seasonal"] = {
            "type": np.random.choice(['sin periodic fluctuation', 'no periodic fluctuation'], p=[0.95, 0.05]),
            "amplitude": random.uniform(attribute_set['seasonal']['amplitude']['min'], attribute_set['seasonal']['amplitude']['max'])
        }
    else:
        description["seasonal"] = {
            "type": np.random.choice(['sin periodic fluctuation', 'no periodic fluctuation'], p=[0.95, 0.05]),
            "amplitude": random.uniform(0, 100)
        }
    # print(description["seasonal"])
    if not ENABLE_MULTIPLE_TREND:
        if "multiple" in attribute_set['trend']['attributes']:
            attribute_set['trend']['attributes'].remove("multiple")
            if len(attribute_set['trend']['attributes']) == 0:
                attribute_set['trend']['attributes'] = ['increase', 'decrease', 'keep steady']
    trend_p = [all_attribute_set['overall_attribute']['trend'][i] for i in attribute_set['trend']['attributes']]
    description["trend"] = {
        "type": np.random.choice(list(attribute_set['trend']['attributes']), p=np.array(trend_p)/sum(trend_p)),
        "start": random.uniform(attribute_set['trend']['start']['min'], attribute_set['trend']['start']['max']),
        "amplitude": random.uniform(attribute_set['trend']['amplitude']['min'], attribute_set['trend']['amplitude']['max'])
    }

    num_local_chars = len(change_positions)
    change_p = [all_attribute_set['change'][i] for i in attribute_set['change']['attributes']]
    local_chars = list(np.random.choice(list(attribute_set['change']['attributes']), size=num_local_chars, p=np.array(change_p)/sum(change_p)))
    
    description["local"] = []
    position_start = attribute_set['change']['position_start'][:num_local_chars]
    for char, position in zip(local_chars, position_start):
        description["local"].append({
            "type": char,
            "position_start": position,
            "amplitude": random.uniform(attribute_set['change']['amplitude']['min'], attribute_set['change']['amplitude']['max'])
        })
    # print(description["local"])
    if 'no periodic fluctuation' not in description["seasonal"]['type']:
        # Generate period and then determine type
        period = random.uniform(attribute_set['seasonal']['period']['min'], attribute_set['seasonal']['period']['max'])
        # if period < 30.0:
        if random.random() < 0.5:
            description["frequency"] = {'type': 'high frequency', 'period': round(period, 1)}
        else:
            description["frequency"] = {'type': 'low frequency', 'period': round(period, 1)}
    else:
        description["frequency"] = {'type': 'no periodicity'}
        
    noise_p = [all_attribute_set['overall_attribute']['noise'][i] for i in attribute_set['noise']['attributes']]
    description["noise"] = {'type': np.random.choice(list(attribute_set['noise']['attributes']), p=np.array(noise_p)/sum(noise_p))}

    return description

def generate_seasonal_wave(period, amplitude_list, split_points, seq_len, wave_type=None):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)
    base_frequency = 1 / period

    # Amplitude series
    amplitude_series = np.zeros(seq_len)
    for i in range(len(amplitude_list)):
        amplitude_series[split_points[i]:split_points[i + 1]] = amplitude_list[i]

    # Smoothing amplitude_series with window size
    sliding_window = 5
    for i in range(seq_len - sliding_window):
        amplitude_series[i + sliding_window // 2] = np.mean(amplitude_series[i:i + sliding_window])

    if wave_type is None:
        wave_type = str(np.random.choice(['sin', 'square', 'triangle'], p=[0.95, 0.03, 0.02]))

    if wave_type == 'sin':
        # num_harmonics = np.random.randint(1, max(2, min(period // 6, 10)))
        num_harmonics = np.random.randint(1, 10)
        for n in range(1, num_harmonics + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            harmonic_amplitude = amplitude_series / n * (1 + np.random.uniform(0, 0.05) * np.sin(np.random.uniform(1, 3) * np.pi * t / seq_len + np.random.uniform(0, 2 * np.pi)))
            data += harmonic_amplitude * np.sin(2 * np.pi * base_frequency * n * t + phase)
    elif wave_type == 'square':
        start = np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.3)
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < start + duration:
                data[i] = amplitude_series[i]
            else:
                data[i] = 0.0
    else:
        start = np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.6)
        end = start + duration
        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < end:
                if cycle_pos < (start + end) / 2:
                    data[i] = amplitude_series[i] * 2 * (cycle_pos - start) / duration
                else:
                    data[i] = amplitude_series[i] * 2 * (end - cycle_pos) / duration
            else:
                data[i] = 0.0

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * max(amplitude_list)
    data -= np.mean(data)

    return data

def generate_sin_noise(amplitude, seq_len):
    # Time array
    t = np.linspace(0, seq_len, seq_len)
    data = np.zeros(seq_len)

    num_harmonics = 200
    for n in range(1, num_harmonics + 1):
        phase = np.random.uniform(0, 2 * np.pi)
        cur_freq = np.random.uniform(50 / seq_len, 200 / seq_len)
        data += np.sin(cur_freq * t + phase) * np.random.uniform(0.3, 1.0)

    # normalize to amplitude
    data = data / (data.max() - data.min() + 1e-7) * amplitude
    data -= np.mean(data)

    return data

def generate_ts_change(length: int, amplitude: float, add_random_noise: bool=True):
    x = np.arange(length) / length
    func = random.choice([
        lambda x: x ** 2,
        lambda x: np.sin(x * np.pi / 2),
        lambda x: x,
        lambda x: 1.0 - (1.0 - x) ** 2
    ])
    cur_value = func(x)

    if add_random_noise:
        # Randomly add noise
        if random.random() > 0.8 and length > 3:
            cur_value += np.random.uniform(-1.0, 1.0, length) * np.random.uniform(0.1, 0.3)

    cur_value = cur_value / (cur_value.max() - cur_value.min() + 1e-7) * amplitude
    return cur_value

def generate_spike(amplitude: float):
    rise_length = np.random.choice([1, 2, 3], p = [0.8, 0.15, 0.05])
    fall_length = np.random.choice([1, 2, 3], p = [0.8, 0.15, 0.05])
    peak_length = np.random.choice([0, 1, 2], p = [0.96, 0.03, 0.01])

    result = np.zeros(rise_length + fall_length + peak_length, dtype=np.float32)
    result[:rise_length] += generate_ts_change(rise_length, amplitude)
    result[rise_length:] += amplitude
    result[rise_length + peak_length:] += generate_ts_change(fall_length, -amplitude)
    
    return result, rise_length, peak_length, fall_length

def generate_noise(attribute_pool, y, overall_amplitude, seq_len):
    max_change = np.abs(np.max(y) - np.min(y))
    noise_level = attribute_pool["noise"]['type']
    if noise_level == "noisy":
        if random.random() > 0.5 and max_change > overall_amplitude / 2 and attribute_pool["frequency"]['type'] == "no periodicity":
            # Generate a sin type noise
            noise = generate_sin_noise(0.2 * overall_amplitude, seq_len)
            noise += np.random.normal(0, 0.03 * overall_amplitude, seq_len)
            std = round(float(np.std(noise)), 3)
            attribute_pool["noise"]["detail"] = f"There is a irregular fluctuating noise, indicating a noisy curve: "
        else:
            # Generate random type noise
            std = np.random.uniform(0.03, 0.15) * overall_amplitude
            noise = np.random.normal(0, std, seq_len)
            attribute_pool["noise"]["detail"] = f"There is a random noise, indicating a noisy curve: "

        # Apply noise segments
        num_noise_segments = 1
        if ENABLE_MULTIPLE_NOISE:
            num_noise_segments = random.randint(1, 3)
        
        # Choose segments to apply noise
        attribute_pool["noise"]["segments"] = []
        noise_segments = generate_split_points(seq_len, num_noise_segments)
        for i in range(num_noise_segments):
            noise_start = noise_segments[i]
            noise_end = noise_segments[i + 1]
            noise_std_amp = np.random.uniform(0.1, 5.0)
            noise[noise_start:noise_end] *= noise_std_amp
            attribute_pool["noise"]["segments"].append({
                "position_start": noise_start,
                "position_end": noise_end,
                "amplitude": round(noise_std_amp * std, 2),
                "description": f"the noise std is {noise_std_amp * std:.2f} between point {noise_start} and point {noise_end}"
            })
            attribute_pool["noise"]["detail"] += f"the noise std is {noise_std_amp * std:.2f} between point {noise_start} and point {noise_end}, "
        attribute_pool["noise"]["detail"] = attribute_pool["noise"]["detail"][:-2] + ". "
    elif noise_level == "almost no noise":
        if max_change > overall_amplitude / 2:
            std = np.random.uniform(0.0, 0.001) * overall_amplitude
        else:
            std = 0.0
        noise = np.random.normal(0, std, seq_len)
        attribute_pool["noise"]["std"] = round(std, 3)
        attribute_pool["noise"]["detail"] = f"The overall noise standard deviation is around {std:.2f}, very small compared the overall change of the curve. The curve is overall smooth with almost no noise. "
    
    return noise

def generate_seasonal(attribute_pool, overall_amplitude, seq_len):
        y = np.zeros(seq_len)
        period = attribute_pool['frequency']['period']
        if "no period" not in attribute_pool["seasonal"]['type'] and period < seq_len:
            if attribute_pool["seasonal"]['type'] == "periodic fluctuation":
                wave_type = None
            else:
                wave_type = attribute_pool["seasonal"]["type"].split(" ")[0]
            if 'amplitude' not in attribute_pool['seasonal']:
                # Many periods of seasonal amplitudes
                num_seasonal = 1
                if ENABLE_MULTIPLE_SEASONAL:
                    num_seasonal = random.randint(1, 2)
                amp = []
                for _ in range(num_seasonal):
                    amp.append(random.uniform(1.0, 2.0) * overall_amplitude)
                split_points = generate_split_points(seq_len, num_seasonal)
            else:
                # Only one period of seasonal amplitudes
                amp = [attribute_pool['seasonal']['amplitude']]
                split_points = [0, seq_len]

            # need to be fixed            
            # period = []
            # for i in num_seasonal:
            #     period.append(attribute_pool['frequency']['poriod'])
            y += generate_seasonal_wave(period, amp, split_points, seq_len, wave_type)

            attribute_pool["seasonal"]['detail'] = f"The time series shows periodicity:"
            attribute_pool["seasonal"]["segments"] = []
            for i in range(len(amp)):
                attribute_pool["seasonal"]["segments"].append({
                    "amplitude": round(amp[i], 2),
                    "position_start": split_points[i],
                    "position_end": split_points[i + 1],
                    "description": f"The amplitude of the periodic fluctuation is {amp[i]:.1f} between point {split_points[i]} and point {split_points[i + 1]}, the period of the fluctuation is {period}."
                })
                attribute_pool["seasonal"]['detail'] += f" The amplitude of the periodic fluctuation between point {split_points[i]} and point {split_points[i + 1]} is {amp[i]:.1f}, the period of the fluctuation is {period}."
            # attribute_pool["seasonal"]['detail'] = attribute_pool["seasonal"]['detail'][:-2] + ". "
        elif attribute_pool["seasonal"]['type'] == "no periodic fluctuation" or period >= seq_len:
            y += 0.0
            attribute_pool["seasonal"]["segments"] = []
            attribute_pool["seasonal"]['detail'] = f"No periodic fluctuations observed."
        return y

def generate_local_chars(attribute_pool, overall_amplitude, seq_len):
    """
    Generate a time series with local characteristics based on the given attribute pool.
    Parameters:
    attribute_pool (dict): A dictionary containing local characteristics to be applied to the time series.
                           Each local characteristic should have a "type", "position_start", and "amplitude".
    overall_amplitude (float): The overall amplitude to scale the local characteristics.
    seq_len (int): The length of the time series to be generated.
    Returns:
    np.ndarray: A numpy array representing the generated time series with local characteristics applied.
    Notes:
    - The function applies various types of local characteristics such as "shake", "upward spike", "downward spike",
      "continuous upward spike", "continuous downward spike", "upward convex", "downward convex", "sudden increase",
      "sudden decrease", "rapid rise followed by slow decline", "slow rise followed by rapid decline",
      "rapid decline followed by slow rise", "slow decline followed by rapid rise", "decrease after upward spike",
      "increase after downward spike", "increase after upward spike", "decrease after downward spike",
      "wide upward spike", and "wide downward spike".
    - The function uses several magic numbers for parameters such as peak lengths, amplitude scaling factors, and
      random ranges. These values are set empirically based on the desired characteristics of the generated time series.
    """
    x = np.arange(seq_len)
    y = np.zeros(seq_len)

    # Apply local attribute
    for local_char in attribute_pool["local"]:
        local_position = local_char["position_start"]
        local_amplitude = local_char["amplitude"]
        if local_position is None:
            while True:
                local_position = random.randint(1, seq_len - 5)

                if all([abs(local_position - i['position_start']) > seq_len / 8 for i in attribute_pool['local'] if i['position_start'] is not None]):
                    local_char['position_start'] = local_position
                    break
        if local_char["type"] == "shake":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            peak_start = local_position
            peak_length = random.randint(6, int(seq_len * 0.15))
            func = random.choice([
                lambda x: np.random.uniform(-1, 1, peak_length) * local_amplitude / 2,
                lambda x: np.sin(x[peak_start:peak_start + peak_length] * 5.0) * local_amplitude / 2
            ])
            y[peak_start:peak_start + peak_length] += func(x)
            local_char["detail"] = f"shake with an amplitude of about {local_amplitude:.2f} occurred between {peak_start}th and {peak_start + peak_length}th value."
            local_char['position_end'] = peak_start + peak_length
            local_char['simple'] = 'Shake.'
            local_char['index'] = [int(i) for i in range(peak_start, peak_start + peak_length)]
        elif local_char["type"] == "upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_start = local_position
            spike, r, p, f = generate_spike(local_amplitude)
            peak_length = len(spike)
            y[peak_start:peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            local_char["position_end"] = peak_start + peak_length
            # local_char["detail"] = f"an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, with the time series value rapidly rising from around <|{peak_start}|> to around <|{spike_top_idx}|> and then quickly falling back to around <|{peak_start + peak_length}|>"
            local_char["detail"] = f"The differences between {peak_start}th and {peak_start + r}th points are much larger than others, indicating there is a sudden increase. The differences between {peak_start + r + p}th and {peak_start + r + p + f}th points are much smaller than others, indicating there is a sudden decrease. Sudden decrease after sudden increase to indicate there is an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, with the time series value rapidly rising from around <|{peak_start}|> to around <|{spike_top_idx}|> and then quickly falling back to around <|{peak_start + peak_length}|>."
            local_char['simple'] = f"Upward spike between {peak_start}th and {peak_start + r +p +f}th points with an amplitude of {local_amplitude:.2f}."
            local_char['index'] = [int(spike_top_idx)]
        elif local_char["type"] == "downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_start = local_position
            spike, r, p, f = generate_spike(-local_amplitude)
            peak_length = len(spike)
            y[peak_start:peak_start + peak_length] += spike
            local_char["position_end"] = peak_start + peak_length
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            local_char["detail"] = f"The differences between {peak_start }th and {peak_start  + r}th points are much smaller than others, indicating there is a sudden decrease. The differences between {peak_start  + r + p}th and {peak_start + r + p + f}th points are much larger than others, indicating there is a sudden increase. Sudden increase after sudden decrease to indicate there is a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start - 1} and point {peak_start - 1 + peak_length}, with the time series value rapidly falling from around <|{peak_start}|> to around <|{spike_top_idx}|> and then quickly rising back to around <|{peak_start + peak_length}|>."
            # local_char["detail"] = f"a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, with the time series value rapidly falling from around {peak_start}th to around {spike_top_idx}th value and then quickly rising back to around {peak_start + peak_length}th value"
            local_char['simple'] = f"Downward spike between {peak_start }th and {peak_start  + r +p +f}th points with an amplitude of {local_amplitude:.2f}."
            local_char['index'] = [int(spike_top_idx)]
        elif local_char["type"] == "continuous upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_region_start = local_position
            num_peaks = random.randint(2, 5)
            peaks = []
            spike_top_ids = []
            all_amplitudes = []
            for _ in range(num_peaks):
                peak_start = random.randint(peak_region_start + 0, peak_region_start + 5)
                cur_amplitude = random.uniform(local_amplitude * 0.6, local_amplitude * 1.5)
                all_amplitudes.append(cur_amplitude)
                peaks.append(f"point {peak_start}")
                spike, r, p, f = generate_spike(cur_amplitude)
                peak_length = len(spike)
                # local_char["position_end"] = max(peak_start + peak_length, local_char["position_end"])
                y[peak_start:peak_start + peak_length] += spike
                peak_region_start = peak_start + peak_length
                spike_top_ids.append(peak_start + np.argmax(np.abs(spike)))
            
            local_char["position_end"] = peak_start + peak_length
            local_amplitude = float(np.mean(all_amplitudes))
            local_char["detail"] = f"The differences shows there are many sudden decreases after sudden increases. Therefore, at {' and '.join(peaks)}, there were {num_peaks} consecutive upward spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly rising sharply from around <|{local_position}|> to around <|" + '|> and <|'.join(map(str, spike_top_ids)) + f"|>, and then quickly falling back to around <|{peak_start + peak_length}|>."
            # local_char["detail"] = f"The differences shows that there are many sudden decreases after sudden increases. Therefore, at {' and '.join(peaks)}, there were {num_peaks} consecutive upward spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly rising sharply from around {local_position}th value to around " + 'th and '.join(map(str, spike_top_ids)) + f"th values, and then quickly falling back to around {peak_start + peak_length}th value."
            local_char['simple'] = f"Continuous upward spike between {peak_region_start }th and {local_char['position_end']}th points with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}."
            local_char['index'] = [int(i) for i in spike_top_ids]
        elif local_char["type"] == "continuous downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            peak_region_start = local_position
            num_peaks = random.randint(2, 5)
            peaks = []
            all_amplitudes = []
            spike_top_ids = []
            for _ in range(num_peaks):
                peak_start = random.randint(peak_region_start + 0, peak_region_start + 5)
                cur_amplitude = random.uniform(local_amplitude * 0.6, local_amplitude * 1.5)
                all_amplitudes.append(cur_amplitude)
                peaks.append(f"point {peak_start}")
                spike , r, p, f = generate_spike(-cur_amplitude)
                peak_length = len(spike)
                # local_char["position_end"] = max(peak_start + peak_length, local_char["position_end"])
                y[peak_start:peak_start + peak_length] += spike
                peak_region_start = peak_start + peak_length
                spike_top_ids.append(peak_start + np.argmax(np.abs(spike)))
            
            local_char["position_end"] = peak_start + peak_length
            local_amplitude = float(np.mean(all_amplitudes))
            local_char["detail"] = f"The differences shows there are many sudden increases after sudden decreases. Therefore, at {' and '.join(peaks)}, there were {num_peaks} consecutive downward spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly falling sharply from around <|{local_position - 1}|> to around <|" + '|> and <|'.join(map(str, spike_top_ids)) + f"|>, and then quickly rising back to around <|{peak_start + peak_length}|>."
            # local_char["detail"] = f"The differences shows that there are many sudden increases after suddent decreases. Therefore, at {' and '.join(peaks)}, there were {num_peaks} consecutive downward spikes with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}, with the time series value repeatedly falling sharply from around {local_position }th value to around " + 'th and '.join(map(str, spike_top_ids)) + f"th values, and then quickly rising back to around {peak_start + peak_length}th value."
            local_char['simple'] = f"Continuous downward spike between {peak_region_start}th and {local_char['position_end']}th points with amplitudes ranging from {min(all_amplitudes):.2f} to {max(all_amplitudes):.2f}."
            local_char['index'] = [int(i) for i in spike_top_ids]
        elif local_char["type"] == "upward convex":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            convex_start = local_position
            start_length, end_length = random.randint(1, 5), random.randint(1, 5)
            convex_length = random.randint(int(seq_len * 0.03), int(seq_len * 0.2))
            y[convex_start:convex_start + start_length] += generate_ts_change(start_length, local_amplitude)
            y[convex_start + start_length:convex_start + start_length + convex_length + end_length] += local_amplitude
            y[convex_start + start_length + convex_length: convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, -local_amplitude)
            local_char["position_end"] = convex_start + start_length + convex_length + end_length

            # Add some noise
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * local_amplitude

            local_char["detail"] = f"The differences shows that the time series increases from {convex_start}th to {convex_start + start_length}th points (with the values rise from <|{convex_start}|> to <|{convex_start + start_length}|>) and keeps raletively stable from {convex_start + start_length}th to {convex_start  + start_length + convex_length}th values and then decreases from {convex_start + start_length + convex_length}th to {convex_start  + start_length + convex_length + end_length}th points (with the values drop to around <|{convex_start + start_length + convex_length + end_length}|>). Therefore the time series forms an upward convex starting from {convex_start}th point with an amplitude of about {local_amplitude:.2f}, and then falls back to normal around {convex_start + start_length + convex_length + end_length}th point."
            local_char['simple'] = f"An upward convex starting from {convex_start}th point with an amplitude of about {local_amplitude:.2f}, and then falls back to normal around {convex_start + start_length + convex_length + end_length }th point."
            # local_char['index'] = [int(i) for i in range(convex_start + start_length, convex_start + start_length + convex_length)]
            local_char['index'] = [int(convex_start + start_length + np.argmax(np.abs(y[convex_start + start_length: convex_start + start_length + convex_length])))]
        elif local_char["type"] == "downward convex":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            convex_start = local_position
            start_length, end_length = random.randint(1, 5), random.randint(1, 5)
            convex_length = random.randint(int(seq_len * 0.03), int(seq_len * 0.2))
            y[convex_start:convex_start + start_length] += generate_ts_change(start_length, -local_amplitude)
            y[convex_start + start_length:convex_start + start_length + convex_length + end_length] -= local_amplitude
            y[convex_start + start_length + convex_length: convex_start + start_length + convex_length + end_length] += generate_ts_change(end_length, local_amplitude)
            local_char["position_end"] = convex_start + start_length + convex_length + end_length

            # Add some noise
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.sin((0.8 + np.abs(random.normalvariate(0.0, 2.0))) * x)[convex_start + start_length:convex_start + start_length + convex_length]
            if random.random() > 0.7:
                y[convex_start + start_length:convex_start + start_length + convex_length] += np.random.uniform(-1.0, 1.0, convex_length) * np.random.uniform(0.1, 0.5) * local_amplitude

            local_char["detail"] = f"The differences shows that the time series decreases from {convex_start}th to {convex_start + start_length}th values (with the values drop from <|{convex_start}|> to <|{convex_start + start_length}|>) and keeps raletively stable from {convex_start + start_length}th to {convex_start + start_length + convex_length}th values and then increases from {convex_start + start_length + convex_length}th to {convex_start + start_length + convex_length + end_length}th values(with the values rise to around <|{convex_start + start_length + convex_length + end_length}|>). Therefore the time series forms a downward convex starting from {convex_start}th value with an amplitude of about {local_amplitude:.2f}, and then rises back to normal around {convex_start + start_length + convex_length + end_length}th value."
            local_char['simple'] = f"An downward convex starting from {convex_start}th value with an amplitude of about {local_amplitude:.2f}, and then rises back to normal around {convex_start + start_length + convex_length + end_length}th value."
            # local_char['index'] = [int(i) for i in range(convex_start + start_length, convex_start + start_length + convex_length)]
            local_char['index'] = [int(convex_start + start_length + np.argmax(abs(y[convex_start + start_length: convex_start + start_length + convex_length])))]
        elif local_char["type"] == "sudden increase":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 10)
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, local_amplitude)
            y[local_position + drop_length:] += local_amplitude
            local_char["position_end"] = local_position + drop_length
            local_char["detail"] = f"The differences between {local_position }th and {local_position + drop_length }th point are much larger than others, which indicates there is a sudden increase with an amplitude of {local_amplitude:.2f} occurred between point {local_position } and point {local_position + drop_length }, with the time series value rising from around <|{local_position - 1}|> to around <|{local_position + drop_length}|>."
            local_char['simple'] = f"Sudden increase between {local_position }th and {local_position + drop_length }th point."
            # local_char['index'] = [int(i) for i in range(local_position, local_position + drop_length)]
            local_char['index'] = [int(local_position + drop_length - 1)]
            if random.random() < 0:
                recover_length = random.randint(1, 10.0)
                recover_amplitude = random.uniform(0, local_amplitude / 3)
                y[local_position + drop_length: local_position + drop_length + recover_length] += generate_ts_change(recover_length, -recover_amplitude)
                y[local_position + drop_length + recover_length:] -= recover_amplitude

                if ENABLE_DROP_PROMPT:
                    local_char["detail"] += f", then a drop with an amplitude of {recover_amplitude:.2f} occurred between point {local_position + drop_length} and point {local_position + drop_length + recover_length}, with the time series value falling back to around <|{local_position + drop_length + recover_length + 1}|>"
        elif local_char["type"] == "sudden decrease":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 10)
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y[local_position + drop_length:] += -local_amplitude
            local_char["position_end"] = local_position + drop_length
            local_char["detail"] = f"The differences between {local_position }th and {local_position + drop_length}th point are much smaller than others, which indicatesa sudden decrease with an amplitude of {local_amplitude:.2f} occurred between point {local_position } and point {local_position + drop_length}, with the time series value falling from around <|{local_position - 1}|> to around <|{local_position + drop_length}|>.."
            local_char['simple'] = f"Sudden decrease between {local_position }th and {local_position + drop_length}th point."
            # local_char['index'] = [int(i) for i in range(local_position, local_position + drop_length)]
            local_char['index'] = [int(local_position + drop_length - 1)]
            if random.random() < 0:
                recover_length = random.randint(1, 10.0)
                recover_amplitude = random.uniform(0, local_amplitude / 3)
                y[local_position + drop_length: local_position + drop_length + recover_length] += generate_ts_change(recover_length, recover_amplitude)
                y[local_position + drop_length + recover_length:] += recover_amplitude

                if ENABLE_DROP_PROMPT:
                    local_char["detail"] += f", then a rise with an amplitude of {recover_amplitude:.2f} occurred between point {local_position + drop_length} and point {local_position + drop_length + recover_length}, with the time series value rising back to around <|{local_position + drop_length + recover_length + 1}|>"
        elif local_char["type"] == "rapid rise followed by slow decline":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            rise_length = random.randint(1, 5)  # Length of rapid rise
            fall_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow decline
            y[local_position: local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y[local_position + rise_length: local_position + rise_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + fall_length
            # local_char["detail"] = (
            #     f"The differences indicate that there is a rapid rise with an amplitude of {local_amplitude:.2f} occurred between point {local_position } and point {local_position + rise_length }, "
            #     f"with the time series value rising from around {local_position }th to around {local_position + rise_length}th points, "
            #     f"followed by a slow decline between point {local_position + rise_length } and point {local_position + rise_length + fall_length}."
            # )
            local_char["detail"] = (
                f"The differences indicate that there isa rapid rise with an amplitude of {local_amplitude:.2f} occurred between point {local_position} and point {local_position + rise_length}, "
                f"with the time series value rising from around <|{local_position - 1}|> to around <|{local_position + rise_length}|>, "
                f"followed by a slow decline between point {local_position + rise_length} and point {local_position + rise_length + fall_length} back to around <|{local_position + rise_length + fall_length}|>"
            )
            local_char['simple'] = f"Rapid rise followed by slow decline between {local_position }th and {local_char['position_end']}th points."
            local_char['index'] = [int(local_position + rise_length - 1)]
        elif local_char["type"] == "slow rise followed by rapid decline":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            rise_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow rise
            fall_length = random.randint(1, 5)  # Length of rapid decline
            y[local_position: local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            y[local_position + rise_length: local_position + rise_length + fall_length] += generate_ts_change(fall_length, -local_amplitude) + local_amplitude
            local_char["position_end"] = local_position + rise_length + fall_length
            # local_char["detail"] = (
            #     f"The differences indicate that starting from point {local_position }, the time series value slowly rises, "
            #     f"reaching a peak at point {local_position + rise_length }, followed by a rapid decline between point {local_position + rise_length } and point {local_position + rise_length + fall_length }."
            # )
            local_char["detail"] = (
                f"The differences indicate that starting from point {local_position}, the time series value slowly rises, "
                f"reaching a peak at point {local_position + rise_length}, followed by a rapid decline between point {local_position + rise_length} and point {local_position + rise_length + fall_length} back to around <|{local_position + rise_length + fall_length}|>"
            )
            local_char['simple'] = f"Slow rise followed by rapid decline between {local_position }th and {local_char['position_end']}th points."
            local_char['index'] = [int(local_position + rise_length - 1)]
        elif local_char["type"] == "rapid decline followed by slow rise":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(1, 5)  # Length of rapid decline
            rise_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow rise
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y[local_position + drop_length: local_position + drop_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + rise_length
            # local_char["detail"] = (
            #     f"The differences indicate that there is a rapid decline with an amplitude of {local_amplitude:.2f} occurred between point {local_position } and point {local_position + drop_length }, "
            #     f"with the time series value falling from around {local_position}th to around {local_position + drop_length}th value, "
            #     f"followed by a slow rise between point {local_position + drop_length } and point {local_position + drop_length + rise_length }."
            # )
            local_char["detail"] = (
                f"The differences indicate that there is a rapid decline with an amplitude of {local_amplitude:.2f} occurred between point {local_position} and point {local_position + drop_length}, "
                f"with the time series value falling from around <|{local_position - 1}|> to around <|{local_position + drop_length}|>, "
                f"followed by a slow rise between point {local_position + drop_length} and point {local_position + drop_length + rise_length} back to around <|{local_position + drop_length + rise_length}|>"
            )
            local_char['simple'] = f"Rapid decline followed by slow rise between {local_position }th and {local_char['position_end'] }th points."
            local_char['index'] = [int(local_position + rise_length - 1)]
        elif local_char["type"] == "slow decline followed by rapid rise":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 2.0))) * overall_amplitude
            drop_length = random.randint(int(seq_len * 0.05), int(seq_len * 0.15))  # Length of slow decline
            rise_length = random.randint(1, 5)  # Length of rapid rise
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            y[local_position + drop_length: local_position + drop_length + rise_length] += generate_ts_change(rise_length, local_amplitude) - local_amplitude
            local_char["position_end"] = local_position + drop_length + rise_length
            # local_char["detail"] = (
            #     f"The differences indicate that starting from point {local_position }, the time series value slowly declines, "
            #     f"reaching a low point at point {local_position + drop_length}, followed by a rapid rise between point {local_position + drop_length } and point {local_position + drop_length + rise_length}."
            # )
            local_char["detail"] = (
                f"The differences indicate that starting from point {local_position}, the time series value slowly declines, "
                f"reaching a low point at point {local_position + drop_length}, followed by a rapid rise between point {local_position + drop_length} and point {local_position + drop_length + rise_length} back to around <|{local_position + drop_length + rise_length}|>"
            )
            local_char['simple'] = f"Slow decline followed by rapid rise between {local_position - 1}th and {local_char['position_end'] - 1}th points."
            local_char['index'] = [int(local_position + rise_length - 1)]
        elif local_char["type"] == "decrease after upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike , r, p, f = generate_spike(local_amplitude)
            peak_length = len(spike)
            fall_length = random.randint(2, int(seq_len * 0.05))  # Length of slow decline
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + fall_length] += generate_ts_change(fall_length, -fall_amplitude)
            y[peak_start + peak_length + fall_length: ] -= fall_amplitude
            local_char["position_end"] = peak_start + peak_length + fall_length
            # local_char["detail"] = (
            #     f"The differences indicate that there is an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start } and point {peak_start + peak_length }, "
            #     f"with the time series value rapidly rising from around {peak_start }th to around {spike_top_idx}th value and quickly falling back, "
            #     f"followed by a further decline between point {peak_start + peak_length } and point {peak_start + peak_length + fall_length }."
            # )
            local_char["detail"] = (
                f"The differences indicate that there is an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly rising from around <|{peak_start - 1}|> to around <|{spike_top_idx}|> and quickly falling back, "
                f"followed by a further decline between point {peak_start + peak_length} and point {peak_start + peak_length + fall_length} to around <|{peak_start + peak_length + fall_length}|>"
            )
            local_char['simple'] = f"Decrease after upward spike between {local_position }th and {local_char['position_end'] }th points."
            local_char['index'] = [int(spike_top_idx)]
        elif local_char["type"] == "increase after downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike , r, p, f = generate_spike(-local_amplitude)
            peak_length = len(spike)
            rise_length = random.randint(2, int(seq_len * 0.05))  # Length of slow rise
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + rise_length] += generate_ts_change(rise_length, fall_amplitude)
            y[peak_start + peak_length + rise_length: ] += fall_amplitude
            local_char["position_end"] = peak_start + peak_length + rise_length
            # local_char["detail"] = (
            #     f"The differences indicate that there is a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start - 1} and point {peak_start + peak_length - 1}, "
            #     f"with the time series value rapidly falling from around {peak_start }th to around {spike_top_idx}th values and quickly rising back, "
            #     f"followed by a further rise between point {peak_start + peak_length } and point {peak_start + peak_length + rise_length }."
            # )
            local_char["detail"] = (
                f"The differences indicate that there is a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly falling from around <|{peak_start}|> to around <|{spike_top_idx}|> and quickly rising back, "
                f"followed by a further rise between point {peak_start + peak_length} and point {peak_start + peak_length + rise_length} to around <|{peak_start + peak_length + rise_length}|>"
            )
            local_char['simple'] = f"Increase after downward spike between {local_position }th and {local_char['position_end'] }th points."
            local_char['index'] = [int(spike_top_idx)]
        elif local_char["type"] == "increase after upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike , r, p, f = generate_spike(local_amplitude)
            peak_length = len(spike)
            fall_length = random.randint(2, int(seq_len * 0.05))  # Length of slow decline
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + fall_length] += generate_ts_change(fall_length, fall_amplitude)
            y[peak_start + peak_length + fall_length: ] += fall_amplitude
            local_char["position_end"] = peak_start + peak_length + fall_length
            # local_char["detail"] = (
            #     f"The differences indicate that there is an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start } and point {peak_start + peak_length }, "
            #     f"with the time series value rapidly rising from around {peak_start }th to around {spike_top_idx}th values and quickly falling back, "
            #     f"followed by a further rise between point {peak_start + peak_length } and point {peak_start + peak_length + fall_length }."
            # )
            local_char["detail"] = (
                f"The differences indicate that there is an upward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly rising from around <|{peak_start - 1}|> to around <|{spike_top_idx}|> and quickly falling back, "
                f"followed by a further rise between point {peak_start + peak_length} and point {peak_start + peak_length + fall_length} to around <|{peak_start + peak_length + fall_length}|>"
            )
            local_char['simple'] = f"Increase after upward spike between {local_position }th and {local_char['position_end'] }th points."
            local_char['index'] = [int(spike_top_idx)]
        elif local_char["type"] == "decrease after downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            fall_amplitude = random.uniform(0.1, 0.7) * local_amplitude
            peak_start = local_position
            spike, r, p, f = generate_spike(-local_amplitude)
            peak_length = len(spike)
            rise_length = random.randint(2, int(seq_len * 0.05))  # Length of slow rise
            y[peak_start: peak_start + peak_length] += spike
            spike_top_idx = peak_start + np.argmax(np.abs(spike))
            y[peak_start + peak_length: peak_start + peak_length + rise_length] += generate_ts_change(rise_length, -fall_amplitude)
            y[peak_start + peak_length + rise_length: ] -= fall_amplitude
            local_char["position_end"] = peak_start + peak_length + rise_length
            # local_char["detail"] = (
            #     f"The differences indicate that there is a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
            #     f"with the time series value rapidly falling from around {peak_start }th to around {spike_top_idx}th values and quickly rising back, "
            #     f"followed by a further decline between point {peak_start + peak_length } and point {peak_start + peak_length + rise_length}."
            # )
            local_char["detail"] = (
                f"The differences indicate that there is a downward spike with an amplitude of {local_amplitude:.2f} occurred between point {peak_start} and point {peak_start + peak_length}, "
                f"with the time series value rapidly falling from around <|{peak_start}|> to around <|{spike_top_idx}|> and quickly rising back, "
                f"followed by a further decline between point {peak_start + peak_length} and point {peak_start + peak_length + rise_length} to around <|{peak_start + peak_length + rise_length}|>"
            )
            local_char['simple'] = f"Decrease after downward spike between {local_position }th and {local_char['position_end'] }th points."
            local_char['index'] = [int(spike_top_idx)]
        elif local_char["type"] == "wide upward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            # Define longer rise and fall lengths
            rise_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow rise
            peak_length = random.randint(1, 4)  # Length of short peak
            fall_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow decline

            # Slow rise
            y[local_position: local_position + rise_length] += generate_ts_change(rise_length, local_amplitude)
            # Short peak
            y[local_position + rise_length: local_position + rise_length + peak_length + fall_length] += local_amplitude
            # Slow decline
            y[local_position + rise_length + peak_length: local_position + rise_length + peak_length + fall_length] += generate_ts_change(fall_length, -local_amplitude)
            local_char["position_end"] = local_position + rise_length + peak_length + fall_length
            local_char["detail"] = (
                f"a slow rise from around <|{local_position - 1}|> to around <|{local_position + rise_length}|> occurred between point {local_position} and point {local_position + rise_length}, "
                f"forming a short peak with an amplitude of {local_amplitude:.2f}, "
                f"followed by a slow decline between point {local_position + rise_length + peak_length} and point {local_position + rise_length + peak_length + fall_length} back to around <|{local_position + rise_length + peak_length + fall_length + 1}|>"
            )
            local_char['simple'] = f"Wide upward spike between {local_position - 1}th and {local_position + rise_length + peak_length + fall_length}th points."
            # local_char['index'] = [int(i) for i in range(local_position + rise_length, local_position + rise_length + peak_length)]
            local_char['index'] = [int(local_position + rise_length + np.argmax(np.abs(y[local_position + rise_length: local_position + rise_length + peak_length])))]
        elif local_char["type"] == "wide downward spike":
            if local_amplitude is None:
                local_amplitude = (0.8 + np.abs(random.normalvariate(0.0, 6.0))) * overall_amplitude
            # Define longer decline and rise lengths
            drop_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow decline
            peak_length = random.randint(1, 4)  # Length of short trough
            rise_length = random.randint(int(seq_len * 0.02), int(seq_len * 0.08))  # Length of slow rise

            # Slow decline
            y[local_position: local_position + drop_length] += generate_ts_change(drop_length, -local_amplitude)
            # Short trough
            y[local_position + drop_length: local_position + drop_length + peak_length + rise_length] -= local_amplitude
            # Slow rise
            y[local_position + drop_length + peak_length: local_position + drop_length + peak_length + rise_length] += generate_ts_change(rise_length, local_amplitude)
            local_char["position_end"] = local_position + drop_length + peak_length + rise_length
            local_char["detail"] = (
                f"a slow decline from around <|{local_position - 1}|> to around <|{local_position + drop_length}|> occurred between point {local_position} and point {local_position + drop_length}, "
                f"forming a short trough with an amplitude of {local_amplitude:.2f}, "
                f"followed by a slow rise between point {local_position + drop_length + peak_length} and point {local_position + drop_length + peak_length + rise_length} back to around <|{local_position + drop_length + peak_length + rise_length + 1}|>"
            )
            local_char['simple'] = f"Wide downward spike between {local_position - 1}th and {local_position + rise_length + peak_length + rise_length}th points."
            # local_char['index'] = [int(i) for i in range(local_position + rise_length, local_position + rise_length + peak_length)]
            local_char['index'] = [int(local_position + rise_length + np.argmax(np.abs(y[local_position + rise_length: local_position + rise_length + peak_length])))]

        local_char['amplitude'] = local_amplitude

    # Sort the local char
    attribute_pool["local"] = sorted(attribute_pool["local"], key=lambda x: x['position_start'])

    return y

def generate_trend(attribute_pool, y, overall_amplitude, overall_bias, seq_len):
    # Apply trend attribute
    trend = attribute_pool["trend"]["type"]

    if 'amplitude' in attribute_pool['trend']:
        amplitude = attribute_pool['trend']['amplitude']
    else:
        amplitude = random.uniform(0.8, 3.0) * overall_amplitude
    if 'start' in attribute_pool['trend']:
        bias = attribute_pool['trend']['start']
    else:
        bias = overall_bias

    if trend == "decrease":
        cur_value = generate_ts_change(seq_len, -amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is decreasing. "
        attribute_pool["trend"]["trend_list"] = [("decrease", 0, seq_len - 1)]
    elif trend == "increase":
        cur_value = generate_ts_change(seq_len, amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is increasing. "
        attribute_pool["trend"]["trend_list"] = [("increase", 0, seq_len - 1)]
    elif trend == "multiple":
        # Ensure the generated trend has more than one type
        while True:
            points = generate_random_points(seq_len=seq_len)[0]
            if len(generate_trend_list(points, seq_len)) > 1:
                break
        trend_ts = generate_trend_curve(seq_len=seq_len, points=points)[1]
        y += trend_ts * amplitude
        attribute_pool["trend"]["detail"] = "From the perspective of the slope, the overall trend contains multiple different segments: " + generate_trend_prompt(points)
        attribute_pool["trend"]["trend_list"] = generate_trend_list(points, seq_len)
    elif trend == "keep steady":
        y += bias
        attribute_pool["trend"]["detail"] = f"From the perspective of the slope, the overall trend is steady. "
        attribute_pool["trend"]["trend_list"] = [("keep steady", 0, seq_len - 1)]

    # Find increase or decrease in local char
    local_phase_change = [i['type'] for i in attribute_pool["local"] if 'increase' in i['type'] or 'decrease' in i['type']]
    if len(local_phase_change):
        attribute_pool["trend"]["detail"] += f"Local phase changes were observed, including: {', '.join(local_phase_change)}."
    attribute_pool["trend"]["start"] = round(float(y[0]), 2)
    attribute_pool["trend"]["amplitude"] = round(float(y[-1] - y[0]), 2)
    # attribute_pool["trend"]["detail"] += f"The value of time series starts from around {float(y[0]):.2f} and ends at around {float(y[-1]):.2f}, with an overall amplitude of {float(y[-1] - y[0]):.2f}. "
    return y

def generate_split_points(seq_len: int, num_segments: int) -> list:
    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")
    if seq_len < num_segments:
        raise ValueError("Sequence length must be at least equal to the number of segments.")

    min_segment_len = seq_len / num_segments / 2  # Minimum segment length
    split_points = [0]  # Start with the first point
    
    for _ in range(num_segments - 1):
        # Determine the valid range for the next split point
        min_point = split_points[-1] + min_segment_len
        max_point = seq_len - (num_segments - len(split_points)) * min_segment_len
        if min_point >= max_point:
            raise ValueError("Cannot generate split points satisfying the constraints.")
        
        # Randomly select a split point within the valid range
        split_points.append(int(random.uniform(min_point, max_point)))
    split_points.append(seq_len)
    
    return split_points

def generate_time_series(attribute_pool, seq_len=512):
    """
    Generate a time series based on the given attribute pool and sequence length.
    Parameters:
    attribute_pool (dict): A dictionary containing attributes that define the characteristics of the time series.
    seq_len (int, optional): The length of the generated time series. Default is 512.
    Returns:
    tuple: A tuple containing the generated time series (numpy array) and the updated attribute pool (dict).
    The attribute_pool dictionary can contain the following keys:
    - "seasonal": A dictionary with a "type" key that defines the type of seasonal pattern.
    - "trend": A dictionary with a "type" key that defines the type of trend.
    - "frequency": A dictionary with "type" and "period" keys that define the frequency characteristics.
    - "overall_amplitude": A float that defines the overall amplitude of the time series.
    - "overall_bias": A float that defines the overall bias of the time series.
    - "local": A list of dictionaries that define local characteristics of the time series.
    - "statistics": A dictionary that will be populated with statistical information about the generated time series.
    The function performs the following steps:
    1. Adapts to legacy behavior by modifying the attribute pool based on certain conditions.
    2. Generates the base time series using a linear space.
    3. Adjusts the period based on the frequency attribute.
    4. Sets an overall amplitude and bias for the time series.
    5. Applies seasonal attributes to the time series.
    6. Applies local changes to the time series.
    7. Applies trend attributes to the time series.
    8. Replaces details in local characteristics with actual values from the time series.
    9. Adds noise to the time series.
    10. Adds statistical information to the attribute pool.
    Returns the generated time series and the updated attribute pool.
    """
    # Adapt to legacy behavior
    if not ENABLE_MULTIPLE_TREND:
        # (Step 1) Remove seasonal type
        if "no period" not in attribute_pool["seasonal"]['type']:
            attribute_pool["seasonal"]["type"] = "periodic fluctuation"

        # (Step 2) Remove multiple trend
        if attribute_pool["trend"]["type"] == "multiple":
            attribute_pool["trend"]["type"] = random.choice(["increase", "decrease", "keep steady"])

    # Generate timeseries
    x = np.linspace(0, 10 * np.pi, seq_len)
    y = np.zeros_like(x)
    
    period = seq_len
    if "frequency" in attribute_pool:
        if period not in attribute_pool["frequency"]:
            if attribute_pool["frequency"]['type'] == "high frequency":
                period = int(random.uniform(10.0, seq_len - 10))
            elif attribute_pool["frequency"]['type'] == "low frequency":
                period = int(random.uniform(seq_len - 10, 1440.0))

        if attribute_pool["frequency"]['type'] == "no periodicity":
            attribute_pool["frequency"]['period'] = 0.0
            attribute_pool["frequency"]['detail'] = "No significant periodic fluctuations observed, overall almost no periodicity. "
        else:
            if period < seq_len:
                attribute_pool["frequency"]['period'] = round(period, 1)
                attribute_pool["frequency"]['detail'] = f"The time series shows periodicity, each fluctuation period is approximately {period:.1f} points."
            else:
                attribute_pool["frequency"]['period'] = round(period, 1)
                attribute_pool["frequency"]['detail'] = "No significant periodic fluctuations observed, overall almost no periodicity. "
    # Set an overall amplitude for all attributes (to ensure the time series is not too flat)
    if 'overall_amplitude' in attribute_pool and 'overall_bias' in attribute_pool:
        overall_amplitude = attribute_pool['overall_amplitude']
        overall_bias = attribute_pool['overall_bias']
    else:
        overall_amplitude_e = np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7], p=[0.2, 0.3, 0.3, 0.05, 0.05, 0.03, 0.03, 0.03, 0.01])
        overall_amplitude = np.random.uniform(10.0 ** (overall_amplitude_e - 1), 10.0 ** (overall_amplitude_e + 1))
        overall_bias = np.random.uniform(-(10.0 ** (overall_amplitude_e + 1)), 10.0 ** (overall_amplitude_e + 1))
        attribute_pool['overall_amplitude'] = overall_amplitude
        attribute_pool['overall_bias'] = overall_bias

    # Apply Seasonal Feature
    y += generate_seasonal(attribute_pool, overall_amplitude, seq_len)
    
    # Apply local changes
    y += generate_local_chars(attribute_pool, overall_amplitude, seq_len)
    # print(attribute_pool['local'])
    # print(y)
    # index = [i for i in range(len(y))]
    # plt.plot(index, y)
    # plt.savefig('./test.png')
    # return None, None
    # Apply trend attribute
    y = generate_trend(attribute_pool, y, overall_amplitude, overall_bias, seq_len)

    # Replace detail in local_chars
    for local_char in attribute_pool["local"]:
        pattern = re.compile(r'<\|(\d+)\|>')

        def replacer(match):
            n = int(match.group(1))
            return f"{y[n]:.2f}"
        local_char['detail'] = pattern.sub(replacer, local_char['detail'])

    # Add noise
    y += generate_noise(attribute_pool, y, overall_amplitude, seq_len)

    # Add statistic information to description
    attribute_pool["statistics"] = {
        "mean": round(float(np.mean(y)), 2),
        "std": round(float(np.std(y)), 2),
        "max": round(float(np.max(y)), 2),
        "min": round(float(np.min(y)), 2),
        "max_pos": int(np.argmax(y)),
        "min_pos": int(np.argmin(y))
    }

    return y, attribute_pool

def attribute_to_text(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True, include_attributes: List[str] = ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'], cot: bool=False, wrong_answer: bool=False, index_number: float=None) -> str:
    """
    Generates a textual description of a time series based on various attributes and attributes.
    Args:
        time_series (np.ndarray): The time series data as a numpy array.
        attribute_pool (dict): A dictionary containing attribute details for the time series.
        generate_values (bool, optional): Deprecated. Use 'statistic' in include_attributes instead. Defaults to True.
        include_attributes (List[str], optional): A list of attributes to include in the description. Defaults to ['length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'].
    Returns:
        str: A detailed textual description of the time series.
    """
    # Adapt to legacy parameters
    if not generate_values and 'statistic' in include_attributes:
        include_attributes.remove('statistic')
    elif generate_values and 'statistic' not in include_attributes:
        include_attributes.append('statistic')

    seq_len = len(time_series)
    segment_mean = [round(np.mean(time_series[i:i+seq_len//32]), 2) for i in range(0, seq_len, seq_len // 32)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    detailed_description = ''
    # if 'length' in include_attributes:
    #     detailed_description += f"The length of the time series is {seq_len}. "
    # if 'trend' in include_attributes:
    #     detailed_description += f"{attribute_pool['trend']['detail']}"
    if 'periodicity' in include_attributes:
        detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type'] and 'frequency' in include_attributes:
        detailed_description += attribute_pool['frequency']['detail']
    # if 'noise' in include_attributes:
    #     detailed_description += attribute_pool['noise']['detail']
    # if 'local' in include_attributes:
    #     if len(attribute_pool["local"]):
    #         detailed_description += 'In terms of local characteristics, ' + ";".join([f"{i['detail']}, forming a {i['type']}" for i in attribute_pool['local']]) + '. '
    #     else:
    #         detailed_description += 'No local characteristics are found. '
    if 'statistic' in include_attributes:
        detailed_description += f"Specific data details: The time series is divided into 32 segments, with the approximate mean values for each {seq_len // 32}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}."


    if 'period' in include_attributes:
        if cot:
            if wrong_answer:
                detailed_description += attribute_pool['seasonal']['detail']
            else:
                detailed_description += f"To analyze the periodicity of this time series, we can identify the periodicity based on its repeating pattern.\n"
                detailed_description += f"# Step 1:Identify Period Patterns\n"
                if len(attribute_pool["seasonal"]["segments"]) > 0:
                    period = int(attribute_pool['frequency']['period'])
                    detailed_description += f"We continue to identify possible recurring patterns by increasing the length from small to large, and find that the recurring pattern is:\n\n{str(list(time_series[:period]))}\n\nThe length of the repeated pattern is {period}.\n"
                    detailed_description += f"# Step 2: Periodicity Length and Periodicity Check\n"
                    detailed_description += f"Although there is some variation in the period, the main trend of different periods remains the same, so we can say that this is a partially periodic series. Although there are slight variations, it is generally periodic. And we confirm the length of the repeated pattern is {period}.\n"
                else:
                    detailed_description += f"We continue to identify possible recurring patterns by increasing the length from small to large, and find that there are none repeated patterns.\n"
                detailed_description += f"# Final Answer\n" + attribute_pool['seasonal']['detail']
        else:
            if wrong_answer:
                pass
            else:
                detailed_description += attribute_pool['seasonal']['detail']

    if 'anomaly' in include_attributes:
        diff =[round(num,3) for num in np.diff(time_series)]
        anomaly_index = []
        if cot:
            if wrong_answer:
                pass
            else:
                detailed_description += f"In order to identify whether there are data anomalies in the time series, I need to calculate the difference between two adjacent points, identify the local pattern of the time series and determine the anomaly. The following are the specific steps.\n"
                detailed_description += f"# Step 1: Calculate the Difference Between Adjacent Points\n"
                
                detailed_description += f"Diff:\n\n{str(diff)}\n\n"
                detailed_description += f"# Step 2: Analyze the Periodicity and Trend\n" + attribute_pool['seasonal']['detail'] + '  \n' + attribute_pool['trend']['detail'] + '\n'
                detailed_description += f"# Step 3: Identify Local Changes\n"
                if len(attribute_pool['local']) == 0:
                    detailed_description += f"According to the differences, there are no local change in the time series.\n"
                    detailed_description += f"# Step 4: Conclusion\nThere are no data anomalies in the time series."
                else:
                    detailed_description += f"According to the differences, there are local change in the time series.\n\n" 
                    for i in range(len(attribute_pool['local'])):
                        detailed_description += f"{str(i+1)}. " + attribute_pool['local'][i]['detail'] + '\n'
                    detailed_description += f"# Step 4: Conclusion\nThere are data anomalies in the time series.\n"
                    for i in range(len(attribute_pool['local'])):
                        detailed_description += f"{str(i+1)}. Index: {attribute_pool['local'][i]['position_start']}~{attribute_pool['local'][i]['position_end']}.  \nLocal Change: " + attribute_pool['local'][i]['simple'] + "  \nMost Abnormal Part (most extreme value in the local change): " + ','.join([str(time_series[i]) for i in attribute_pool['local'][i]['index']]) + '  \nIndex of Most Abnormal Part: ' + ','.join([str(i) for i in attribute_pool['local'][i]['index']]) + '\n'
                        # anomaly_index += [i for i in range(attribute_pool['local'][i]['position_start'],attribute_pool['local'][i]['position_end'] + 1)]
                        anomaly_index += attribute_pool['local'][i]['index']
                    # anomaly_index = sorted(list(anomaly_index))
                    anomaly_index = sorted(set(anomaly_index))
                    detailed_description += f"\nWe only consider the indexes of the most abnormal part of all local changes as the anomalous indexes. Therefore, all anomalous indexes are: {str(anomaly_index)}."
        else:
            if wrong_answer:
                pass
            else:
                detailed_description += f"In order to identify whether there are data anomalies in the time series, I will to calculate the difference between two adjacent points, identify the local pattern of the time series and finally determine the anomaly.\n"
                detailed_description += f"After finishing these steps, result are as follows:\n\n"
                if len(attribute_pool['local']) == 0:
                    detailed_description += f"There are no local change and no data anomalies in the time series.\n"
                else:
                    detailed_description += f"There are data anomalies in the time series.\n"
                    for i in range(len(attribute_pool['local'])):
                        detailed_description += f"{str(i+1)}. Index: {attribute_pool['local'][i]['position_start']}~{attribute_pool['local'][i]['position_end']}.  \nLocal Change: " + attribute_pool['local'][i]['simple'] + "  \nMost Abnormal Part (most extreme value in the local change): " + ','.join([str(time_series[i]) for i in attribute_pool['local'][i]['index']]) + '  \nIndex of Most Abnormal Part: ' + ','.join([str(i) for i in attribute_pool['local'][i]['index']]) + '\n'
                        # anomaly_index += [i for i in range(attribute_pool['local'][i]['position_start'],attribute_pool['local'][i]['position_end'] + 1)]
                        anomaly_index += attribute_pool['local'][i]['index']
                    # anomaly_index = sorted(list(anomaly_index))
                    anomaly_index = sorted(set(anomaly_index))
                    detailed_description += f"\nWe only consider the indexes of the most abnormal part of all local changes as the anomalous indexes. Therefore, all anomalous indexes are: {str(anomaly_index)}."

    if 'diff' in include_attributes:
        diff =[round(num,3) for num in np.diff(time_series)]
        if cot:
            if wrong_answer:
                detailed_description += f"To calculate the difference sequence between two adjacent points of the time series, we can scan the time series sequentially and calculate the difference one by one."
                detailed_description += f"Afert scanning to the end of the time series, we get the difference sequence:\n\n{str(diff)}\n\n"
            else:
                detailed_description += f"To calculate the difference sequence between two adjacent points of the time series, we can scan the time series sequentially and calculate the difference one by one. Here's the step-by-step process:\n"
                detailed_description += f"# Step 1: calculate the Difference\n"
                diff =[round(num, 3) for num in np.diff(time_series)]
                for i in range(min(10, len(time_series) - 1)):
                    detailed_description += f"{str(i)}. {str(time_series[i+1])} - {str(time_series[i])} = {str(round(time_series[i+1]- time_series[i], 3))}\n"
                detailed_description += f"\n......\n"
                detailed_description += f"\nAfert scanning to the end of the time series, we get the difference sequence:\n\n{str(diff)}\n\n"
                detailed_description +=  f"# Step 2: Conclusion\nDifference squence is:\n\n{str(diff)}"
        else:
            if wrong_answer:
                pass
            else:
                detailed_description += f"To calculate the difference sequence between two adjacent points of the time series, we can scan the time series sequentially and calculate the difference one by one."
                detailed_description += f"Afert scanning to the end of the time series, we get the difference sequence:\n\n{str(diff)}\n\n"

    if 'mean' in include_attributes:
        if cot:
            if wrong_answer:
                pass
            else:
                pass
        else:
            if wrong_answer:
                pass
            else:
                pass

    if 'noise' in include_attributes:
        trend = np.convolve(np.concatenate([[time_series[0], time_series[0]],time_series, [time_series[-1], time_series[-1]]]), np.ones(5)/5, mode='valid')
        trend = np.array([round(num, 3) for num in trend])
        noise = time_series - trend
        noise = np.array([round(num, 3) for num in noise])
        noise_mean = round(np.mean(noise),3)
        noise_std = round(np.std(noise),3)
        value_mean = round(np.mean(time_series),3)
        value_std = round(np.std(time_series),3)
        snr = round((value_mean**2 + value_std**2)/(noise_mean**2 + noise_std**2 + 1e-5), 2)
        if cot:
            if wrong_answer:
                detailed_description += attribute_pool['noise']['detail']
            else:
                detailed_description += f"To determine the noise intensity in the time series, we need to separate the underlying trend from the noise. Here's a step-by-step approach to estimate the noise intensity:\n"
                detailed_description += f"# Step 1: Decompose the Time Series and Extract the Trend\n"
                detailed_description += f"We can use a moving average, a smoothing technique (e.g., exponential smoothing) or STL to estimate the trend. For simplicity, let's use a moving average with a small window (5 points) to smooth the data and approximate the trend. To keep the data length consistent, pad the time series with two digits before and after.\n"
                detailed_description += "\n$t_i = (x_{i-2} + x_{i-1} + x_i + x_{i+1} + x_{i+2})/5$\n"
                detailed_description += f"\nTrend is:\n\n{str(list(trend))}\n\n"
                detailed_description += f"# Step 2: Calculate the Noise\nSubtract the trend from the original time series to isolate the noise component:\n\n$Noise = Original Series - Trend$\n\nThe noise is:\n\n{str(list(noise))}\n\n"
                detailed_description += f"# Step 3: Estimate Noise Intensity\nThe noise intensity can be quantified using the standard deviation or variance of the noise component. A higher standard deviation indicates higher noise intensity.\n"
                detailed_description += attribute_pool['noise']['detail']
                # detailed_description += f"The mean and std of noise is {str(noise_mean)}, {str(noise_std)}. The mean and std of the time series is {str(value_mean)}, {str(value_std)}."
                # if snr > 10:
                #     detailed_description += f"The noise intensity is relatively low."
                # elif snr > 5 and snr <=10:
                #     detailed_description += f"The noise intensity is medium."
                # else:
                #     detailed_description += f"The noise intensity is relatively high."
                detailed_description += f"\n# Final Answer\n"
                detailed_description += attribute_pool['noise']['detail']
                # detailed_description +=  f"The mean and std of noise is {str(noise_mean)}, {str(noise_std)}. The mean and std of the time series is {str(value_mean)}, {str(value_std)}."
                # if snr > 10:
                #     detailed_description += f"The noise intensity is relatively low."
                # elif snr > 5 and snr <=10:
                #     detailed_description += f"The noise intensity is medium."
                # else:
                #     detailed_description += f"The noise intensity is relatively high."

        else:
            if wrong_answer:
                if snr > 10:
                    detailed_description += f"The noise intensity is relatively low."
                elif snr > 5 and snr <=10:
                    detailed_description += f"The noise intensity is medium."
                else:
                    detailed_description += f"The noise intensity is relatively high."
            else:
                detailed_description += attribute_pool['noise']['detail']
                # detailed_description += f"The mean and std of noise is {str(noise_mean)}, {str(noise_std)}. The mean and std of the time series is {str(value_mean)}, {str(value_std)}."
                # if snr > 10:
                #     detailed_description += f"The noise intensity is relatively low."
                # elif snr > 5 and snr <=10:
                #     detailed_description += f"The noise intensity is medium."
                # else:
                #     detailed_description += f"The noise intensity is relatively high."

    if 'index' in include_attributes:
        index = []
        for i in range(len(time_series)):
            if time_series[i] == index_number:
                index.append(str(i))
        if cot:
            if wrong_answer:
                detailed_description += f"The index of {str(index_number)} is {', '.join(index)}.\n"
            else:
                detailed_description += f"To find the index of the value {str(index_number)} in the given time series, we can simply count the position of the value in the list. Here's the step-by-step process:\n"
                detailed_description += f"# Step 1: Scan the Time Series\n\n"
                for i in range(min(5,len(time_series))):
                    if time_series[i] == index_number:
                        detailed_description += f"{str(i)}. Index {str(i)}: {str(time_series[i])} == {str(index_number)}\n"
                    else:
                        detailed_description += f"{str(i)}. Index {str(i)}: {str(time_series[i])} != {str(index_number)}\n"
                detailed_description += f"\n......\n"
                detailed_description += f"\nThus, the index of {str(index_number)} is {', '.join(index)}.\n"
                detailed_description += f"# Step 2: Conclusion\n" + f"The index of {str(index_number)} is {', '.join(index)}.\n"
                detailed_description += f"# Final Answer\n" + f"The index of {str(index_number)} is {', '.join(index)}.\n"

        else:
            if wrong_answer:
                detailed_description += f"The index of {str(index_number)} is {str(random.randint(0, len(time_series)))}.\n"
            else:
                detailed_description += f"The index of {str(index_number)} is {', '.join(index)}.\n"

    if 'length' in include_attributes:
        if cot:
            # step 1
            if wrong_answer:
                detailed_description += f'The length of the time series is {str(len(time_series))}.'
            else:
                detailed_description += f'# Step 1: Understanding the Problem\nFirst, I need to determine the length of the given time series. The length of a time series is simply the number of data points it contains. So, my goal is to count how many numbers are in this list.\n'
                # step 2
                detailed_description += f"# Step 2: Counting the Data Points\nTo find the length, I'll count each number in the series one by one.\nLet's proceed with the counting:\n\n"
                for i in range(len(time_series)):
                    detailed_description += f"{str(i)}. {str(time_series[i])}\n"
                detailed_description += f"\nAfter counting each number, I find that there are {str(len(time_series))} data points in the time series.(Start from index 0)\n\n"
                # step 3
                detailed_description += f"# Step3: Verification\nTo ensure accuracy, I'll recount the data points.\n"
                # for i in range(len(time_series)):
                #     detailed_description += f"{str(i)}. {str(time_series[i])}\n"
                detailed_description += f"The recount confirms that there are indeed {str(len(time_series))} data points.\n"
                # step 4
                detailed_description += f"# Step 4: Conclusion\nAfter carefully counting each data point in the time series and verifying the count, I conclude that the length of the given time series is {str(len(time_series))}."
        else:
            if wrong_answer:
                detailed_description += f'The length of the time series is {str(random.choice([x for x in range(len(time_series)+ 100) if x != len(time_series)]))}.'
            else:
                detailed_description += f'The length of the time series is {str(len(time_series))}.'
    
    if 'min' in include_attributes:
        min_value = np.min(time_series)
        if cot:
            if wrong_answer:
                detailed_description += f"The minimum value of the time series is {str(min_value)}."
            else:

                detailed_description += f"# Step 1:Understanding the Problem\nThe task is to find the minimum value in the given time series. The time series is a sequence of numerical values with length {str(len(time_series))}. The minimum value is the smallest number in this list. To find it, I'll scan through the list and identify the smallest value.\n"
                detailed_description += f"# Step 2: Analyzing the Time Series\nTo find the minimum value, I'll follow these steps:\n1. Initialize a variable to store the minimum value. I'll start by assuming the first value in the list is the smallest.\n2. Iterate through the list and compare each value with the current minimum.\n3. Update the minimum value if I find a smaller number.\n4. Verify the result to ensure accuracy.\n"
                detailed_description += f"# Step 3: Finding the Minimum Value\n## Step 3.1: Initialize the Minimum Value\nI'll start by assuming the first value, {str(time_series[0])}, is the smallest.\nmin_value = {str(time_series[0])}\n## Step 3.2: Iterate Through the List\nNow, I'll go through each value in the list and compare it with min_value. If a value is smaller, I'll update min_value.\nHere is the process:  \n"
                min_value = time_series[0]
                detailed_description += f"\n0. {str(time_series[0])}: Current min_value = {str(time_series[0])}\n"
                for i in range(1, min(5, len(time_series))):
                    if time_series[i] >= min_value:
                        detailed_description += f"{str(i)}. {str(time_series[i])}: Not smaller than {str(min_value)}. No update.\n"
                    else:
                        detailed_description += f"{str(i)}. {str(time_series[i])}: Smaller than {str(min_value)}. Update min_value = {str(time_series[i])}.\n"
                        # min_value = time_series[i]
                detailed_description += f"\n......\n"
                detailed_description += f"## Step 3.3: Final Minimum Value\nAfter iterating through the entire list until the {str(len(time_series))}th value, the smallest value found is {str(min_value)}.\n"
                detailed_description += f"# Step 4: Verification\nTo ensure accuracy, I'll double-check the list for any values smaller than {str(min_value)}. Scanning through the list, I confirm that {str(min_value)} is indeed the smallest value.\n"
                detailed_description += f"# Step 5: Conclusion\nAfter carefully analyzing the time series and verifying the result, I conclude that the minimum value in the time series is {str(min_value)}.\n"
                detailed_description += f"# Final Answer\nThe minimum value of the time series is {str(min_value)}."

        else:
            min_value = np.min(time_series)
            if wrong_answer:
                detailed_description += f"The minimum value of the time series is {str(random.choice(list(time_series)))}."
            else:
                detailed_description += f"The minimum value of the time series is {str(min_value)}."

    if 'max' in include_attributes:
        max_value = np.max(time_series)
        if cot:
            if wrong_answer:
                detailed_description += f"The maximum value of the time series is {str(max_value)}."
            else:
                detailed_description += f"# Step 1:Understanding the Problem\nThe task is to find the maximum value in the given time series. The time series is a sequence of numerical values with length {str(len(time_series))}. The maximum value is the largest number in this list. To find it, I'll scan through the list and identify the largest value.\n"
                detailed_description += f"# Step 2: Analyzing the Time Series\nTo find the maximum value, I'll follow these steps:\n1. Initialize a variable to store the maximum value. I'll start by assuming the first value in the list is the largest.\n2. Iterate through the list and compare each value with the current maximum.\n3. Update the maximum value if I find a larger number.\n4. Verify the result to ensure accuracy.\n"
                detailed_description += f"# Step 3: Finding the Maximum Value\n## Step 3.1: Initialize the Maximum Value\nI'll start by assuming the first value, {str(time_series[0])}, is the largest.\nmax_value = {str(time_series[0])}\n## Step 3.2: Iterate Through the List\nNow, I'll go through each value in the list and compare it with max_value. If a value is larger, I'll update max_value.\nHere is the process:  \n"
                max_value = time_series[0]
                detailed_description += f"\n0. {str(time_series[0])}: Current max_value = {str(time_series[0])}\n"
                for i in range(1, min(5, len(time_series))):
                    if time_series[i] <= max_value:
                        detailed_description += f"{str(i)}. {str(time_series[i])}: Not larger than {str(max_value)}. No update.\n"
                    else:
                        detailed_description += f"{str(i)}. {str(time_series[i])}: Larger than {str(max_value)}. Update max_value = {str(time_series[i])}.\n"
                        # max_value = time_series[i]
                detailed_description += f"\n......\n"
                detailed_description += f"## Step 3.3: Final Maximum Value\nAfter iterating through the entire list until the {str(len(time_series))}th value, the largest value found is {str(max_value)}.\n"
                detailed_description += f"# Step 4: Verification\nTo ensure accuracy, I'll double-check the list for any values larger than {str(max_value)}. Scanning through the list, I confirm that {str(max_value)} is indeed the largest value.\n"
                detailed_description += f"# Step 5: Conclusion\nAfter carefully analyzing the time series and verifying the result, I conclude that the maximum value in the time series is {str(max_value)}.\n"
                detailed_description += f"# Final Answer\nThe maximum value of the time series is {str(max_value)}."
        else:
            max_value = np.max(time_series)
            if wrong_answer:
                detailed_description += f"The maximum value of the time series is {str(random.choice(list(time_series)))}."
            else:    
                detailed_description += f"The maximum value of the time series is {str(max_value)}."
    
    if 'trend' in include_attributes:
        if cot:
            if wrong_answer:
                detailed_description += attribute_pool['trend']['detail']
            else:
                time_series = list(time_series)
                detailed_description += f"To determine the trend of the time series, we can analyze the data step by step. Here's the process:\n"
                detailed_description += f"# 1. Visual Inspection\nFirst, let's inspect the time series data. The length of the time series is {str(len(time_series))}, the maximum and minimum are {str(np.max(time_series))} and {str(np.min(time_series))}.\n"
                for local in attribute_pool['local']:
                    detailed_description += f"There is a {local['type']} between {local['position_start']}th and {local['position_end']}th values, which could be outliers or noise.\n"
                detailed_description += f"# 2. Remove Outliers\n"
                if len(attribute_pool['local']) > 0:
                    positions = [0]
                    for local in attribute_pool['local']:
                        detailed_description += f"The {local['type']} between {local['position_start']}th and {local['position_end']}th values are likely outliers."
                        positions += [local['position_start'], local['position_end']]
                    positions.append(-1)
                    parsed_ts = []
                    for i,j in zip(positions[0::2], positions[1::2]):
                        parsed_ts += time_series[i:j]
                    detailed_description += f"Removing these outliers gives a clearer picture of the trend:\n\n{str(list(parsed_ts))}\n"
                else:
                    detailed_description += f"No outliers in the time serise. Go to step 3.\n"
                    parsed_ts = time_series
                detailed_description += f"# 3. Analyze the Periodicity\n" + attribute_pool["seasonal"]['detail'] + '\n'
                if len(attribute_pool["seasonal"]["segments"]) > 0:
                    period = attribute_pool['frequency']['period']
                    index = [i for i in range(len(parsed_ts)) if i % period == 0]
                    period_series = [parsed_ts[i] for i in index]
                    diff = np.diff(period_series)
                    diff = [round(num,3) for num in diff]
                    detailed_description += f"# 4. Calculate the Trend\nAfter removing outliers and analyzing periodicity, we calculate the difference between points separated by one period. The differences between the 0th, Tth, 2Tth ...(T is the period) are as follows:  \n{str(list(diff))}  \n" + 'We analyze the differences and get the result. ' + attribute_pool['trend']['detail'] + '\n'
                else:
                    diff = [round(num, 3) for num in np.diff(parsed_ts)]
                    detailed_description += f"# 4. Calculate the Trend\nAfter removing outliers and analyzing periodicity, we calculate the difference between points:  \n{str(list(diff))}  \n"+ 'We analyze the differences and get the result. ' + attribute_pool['trend']['detail'] + '\n'
                # detailed_description += f"# 4. Verification\nWe calculate the difference between two adjacent values and confirm the trend of the time series.\n"
                detailed_description += f"# Final Answer\n" + attribute_pool['trend']['detail']
        else:
            if wrong_answer:
                detailed_description += f"The time series has a clear {random.choice([t for t in ['keep steady', 'increase', 'decrease'] if t!=attribute_pool['trend']['type']])} trend with minor fluctuations, despite a few outliers that do not affect the overall trend."
            else:    
                detailed_description += attribute_pool['trend']['detail']
    
    if 'local' in include_attributes:
        all_types = ["shake", "upward spike", "downward spike", "continuous upward spike", "continuous downward spike", "upward convex", "downward convex", "sudden increase",
                            "sudden decrease", "rapid rise followed by slow decline", "slow rise followed by rapid decline",
                            "rapid decline followed by slow rise", "slow decline followed by rapid rise", "decrease after upward spike",
                            "increase after downward spike", "increase after upward spike", "decrease after downward spike",
                            "wide upward spike", "wide downward spike"]
        if len(attribute_pool['local']) > 0:
            local = attribute_pool['local'][0]
            # if True:
            # if local['type'] == 'downward spike' or local['type'] == 'upward spike' or local['type'] == 'continuous upward spike' or local['type'] == 'continuous upward spike':
            if cot:
                if wrong_answer:
                    detailed_description += f"There is {local['detail']}"
                else:
                    # detailed_description += f"To determine if there are local changes in the time series, we need to analyze the data for significant deviations or patterns that indicate a local change. A local change could be a sudden drop, spike, or a change in the trend. Here's the step-by-step process to identify such changes:\n"
                    # detailed_description += f"# Step 1: Visual Inspection\n"
                    # detailed_description += f"The time series generally {attribute_pool['trend']['type']}. However there is a {local['type']} between {str(local['position_start'])}th and {str(local['position_end'])}th values. After the {local['type']}, the series recovers and resumes the trend.\n"
                    # detailed_description += f"# Step 2: Calculate Differences\nTo quantify the changes, compute the differences between consecutive values:\nDifference = x_t - x_{{t-1}}\nSignificant differences (positive or negative) indicate local changes.\n"
                    # for i in range(len(time_series) - 1):
                    #     detailed_description += f"{str(i)}. {str(time_series[i+1])} - {str(time_series[i])} = {str(round(time_series[i+1]- time_series[i], 3))}\n"
                    # detailed_description += f"# Step 3: Identify Local Changes\nBased on the differences, we can identify the following local changes:\n1. {local['detail']}.\n"
                    # detailed_description += f"# Step 4: Conclusion\nThere are local changes in the time series:\n{local['detail']}\n"
                    # # detailed_description += f"Final Answer: There is a {local['type']} between {str(local['position_start'])}th and {str(local['position_end'])}th values."
                    # detailed_description += f"Final Answer: There is {local['detail']}."
                    ## fix text
                    detailed_description += f"To determine if there are local changes in the time series, we need to analyze the data for significant deviations or patterns that indicate a local change. A local change could be a sudden drop, spike, or a change in the trend. Here's the step-by-step process to identify such changes:\n"
                    # detailed_description += f"# Step 1: Input Data\n"
                    # detailed_description += f"The time series data provided is:\n\n{str(list(time_series))}\n"
                    detailed_description += f"# Step 1: Calculate Differences Between Consecutive Points\nTo quantify the changes, compute the differences between consecutive values:\n$Difference = x_t - x_{{t-1}}$\n\n"
                    # diff = []
                    diff =[round(num, 3) for num in np.diff(time_series)]
                    for i in range(min(10, len(time_series) - 1)):
                        detailed_description += f"{str(i)}. {str(time_series[i+1])} - {str(time_series[i])} = {str(round(time_series[i+1]- time_series[i], 3))}\n"
                        # diff.append(round(time_series[i+1]- time_series[i], 3))
                    detailed_description += f"......\n"
                    detailed_description += f"\nNow we get the differences:\n\n{str(diff)}\n"
                    detailed_description += f"# Step 2: Analyze Differences and Identify Local Changes\nBased on the differences and original input data, we can identify the following local changes:\n{local['detail']}\n"
                    detailed_description += f"# Step 3: Conclusion\nThere are local changes in the time series:\n{local['simple']}\n"
                    # detailed_description += f"Final Answer: There is a {local['type']} between {str(local['position_start'])}th and {str(local['position_end'])}th values."
                    detailed_description += f"Final Answer: There is {local['simple']}"
            else:
                if wrong_answer:
                    detailed_description += f"Final Answer: There is an {random.choice(all_types)}."
                else:
                    detailed_description += f"There is {local['detail']}"
        
        else:
            if wrong_answer:
                detailed_description += f"There is an {random.choice(all_types)}."
            else:
                detailed_description += f"There is none local change in the time series. The time series {attribute_pool['trend']['type']} over the whole process."

    return detailed_description


def attribute_to_caption(time_series: np.ndarray, attribute_pool: dict, generate_values: bool=True) -> str:
    """
        Compared with text, caption is in a more natural and fluent way that combines the trend with the local flucations.
    """
    seq_len = len(time_series)
    segment_mean = [round(np.mean(time_series[i:i+seq_len//32]), 2) for i in range(0, seq_len, seq_len // 32)]
    max_value = round(np.max(time_series), 2)
    min_value = round(np.min(time_series), 2)

    # Some basic attribute_pool
    detailed_description = ''
    detailed_description += f"The length of the time series is {seq_len}. "
    detailed_description += attribute_pool['seasonal']['detail']
    if "no" not in attribute_pool['seasonal']['type']:
        detailed_description += attribute_pool['frequency']['detail']
    detailed_description += attribute_pool['noise']['detail']

    # Combine the multiple attribute_pool
    detailed_description += "In terms of the trend and changes of this time series: At the beginning, "
    all_local_changes = dict((int(v['position_start']), v) for v in attribute_pool['local'])
    cur_pos = 0
    while True:
        if cur_pos >= seq_len - 1:
            break

        # Find the next local change
        later_changes = sorted(k for k in all_local_changes if k >= cur_pos)
        later_trend = sorted(k[1] for k in attribute_pool["trend"]["trend_list"] if k[1] > cur_pos)
        cur_trend = [k for k in attribute_pool["trend"]["trend_list"] if (k[1] <= cur_pos < k[2])][0]

        if (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] < later_trend[0]) or (len(later_changes) > 0 and len(later_trend) == 0):
            # Later is a change
            nxt_pos = later_changes[0]
            cur_change = [k for k in attribute_pool["local"] if k['position_start'] == nxt_pos][0]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}; "
            detailed_description += f"from point {cur_change['position_start']} to point {cur_change['position_end']}, {cur_change['detail']}, forming a {cur_change['type']}; "
            cur_pos = cur_change['position_end']
        elif (len(later_changes) > 0 and len(later_trend) > 0 and later_changes[0] >= later_trend[0]) or (len(later_trend) > 0 and len(later_changes) == 0):
            # Later is a trend
            nxt_pos = later_trend[0]
            nxt_trend = [k for k in attribute_pool["trend"]["trend_list"] if k[1] == nxt_pos]
            if nxt_pos > cur_pos:
                detailed_description += f"from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}, and then the trend of the time series changes to {nxt_trend[0][0]}; "
            cur_pos = nxt_pos
        else:
            # Later is the end
            nxt_pos = seq_len - 1
            if nxt_pos > cur_pos:
                detailed_description += f"finally, from point {cur_pos} to {nxt_pos}, the time series {cur_trend[0]} with values from {float(time_series[cur_pos]):.2f} to {float(time_series[nxt_pos]):.2f}. "
            break
    
    if generate_values:
        detailed_description += f"Specific data details: The time series is divided into 32 segments, with the approximate mean values for each {seq_len // 32}-point interval being: {segment_mean}. The maximum value of the entire series is {max_value}, and the minimum value is {min_value}, the start value is {float(time_series[0]):.2f}, the end value if {float(time_series[-1]):.2f}. "
        
        # Random choose some points
        for _ in range(5):
            cur_pos = random.choice(list(range(seq_len)))
            detailed_description += f"The value of point {cur_pos} is {float(time_series[cur_pos]):.2f}. "

    return detailed_description

def prompt_to_inference(timeseries: np.ndarray, prompt: str) -> str:
    prompt_list = prompt.split("<ts><ts/>")
    result = prompt_list[0]

    for i in range(len(prompt_list) - 1):
        cur_ts = timeseries[i]
        if type(cur_ts) == np.ndarray:
            cur_ts = cur_ts.tolist()
        cur_ts = [[round(float(v), 4) for v in item] for item in cur_ts]
        result += f"<ts>{cur_ts}<ts/>" + prompt_list[i + 1]

    return result