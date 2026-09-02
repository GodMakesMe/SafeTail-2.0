import numpy as np
import pandas as pd
import random
import pickle
from scipy.stats import weibull_min
from scipy.special import gamma
from agent import get_computation_delay, get_tramission_delay

# Modified safe version of get_propogation_delay
ping_data = pickle.load(open('btp/ping_data.pkl', 'rb'))
def get_propogation_delay(num):
    idx = max(0, min(num - 1, len(ping_data) - 1))
    return random.choice(ping_data[idx])

# Load required datasets
df = pd.read_csv('data.csv')
res_df = pd.read_csv('inst2.csv', header=None, names=range(12))
resolutions = [res_df.iloc[i, 0].split()[3] for i in range(len(res_df))]
yolo_file_size_df = pd.read_csv('filesize.csv')
instance_file_size_df = pd.read_csv('instance/image_file_sizes.csv')
noise_df = pd.read_csv('noise/noise_final_data_19May.csv')

# Parameters
task = 'yolo'  # change to 'instance' or 'noise' as needed
beta = 5  # number of fog nodes
max_bandwidth = 20  # Mbps
required_latency_samples = 500
xi = 1
tau = 3
p_quantile = 0.95


def get_random_resolution():
    res = random.choice(resolutions)
    return int(res.split('x')[0]) * int(res.split('x')[1])

def get_random_file_size():
    if task == 'yolo':
        return yolo_file_size_df['col1'].sample().values[0]
    if task == 'instance':
        return instance_file_size_df['Size (bytes)'].sample().values[0] / 1024
    if task == 'noise':
        return noise_df['file_size'].sample().values[0] / 1024

# Log storage
selected_latencies = []

step = 0
while len(selected_latencies) < required_latency_samples:
    state = {
        'LOAD': df.iloc[step % len(df), 1:beta+1].values.astype(int),
        'BANDWIDTH': max_bandwidth,
        'MESSAGE_SIZE': get_random_file_size()
    }
    if task == 'yolo':
        state['RESOLUTION'] = get_random_resolution()
    if 'PROPOGATION' not in state:
        state['PROPOGATION'] = [get_propogation_delay(i) for i in state['LOAD']]

    node_costs = []
    node_latency_samples = []

    for node in range(beta):
        samples = []
        for _ in range(100):
            comp = get_computation_delay(state, node, task)
            tran = get_tramission_delay(state['MESSAGE_SIZE'],
                                        state['BANDWIDTH']/state['LOAD'][node],
                                        state['BANDWIDTH']/state['LOAD'][node])
            prop = get_propogation_delay(state['LOAD'][node])
            total = float(comp) + tran + prop
            samples.append(total)

        # Weibull fitting
        shape_k, loc, scale_lambda = weibull_min.fit(samples, floc=0)
        delta_n = weibull_min.ppf(p_quantile, shape_k, loc=0, scale=scale_lambda)
        mean_latency = scale_lambda * gamma(1 + 1 / shape_k)
        cost_n = xi * mean_latency + tau * delta_n

        node_costs.append((node, cost_n))
        node_latency_samples.append((node, samples))

    # Select node with minimum cost
    best_node = min(node_costs, key=lambda x: x[1])[0]

    # Record one random latency sample from selected node
    latency_value = random.choice(node_latency_samples[best_node][1])
    selected_latencies.append({
        'step': step,
        'selected_node': best_node,
        'latency': latency_value
    })

    print(f"Step {step}: Node {best_node}, latency = {latency_value:.4f}")
    step += 1

# Save results
latency_df = pd.DataFrame(selected_latencies)
latency_df.to_csv(f'tlora_selected_latencies_{task}.csv', index=False)
print(f"Saved {len(latency_df)} latency samples to tlora_selected_latencies_{task}.csv")
