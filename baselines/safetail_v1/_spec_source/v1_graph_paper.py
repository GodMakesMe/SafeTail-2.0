import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import ast
from matplotlib.ticker import FixedLocator, FixedFormatter

directory_path = '/home/iiitd/Desktop/aman/ut_dqn/yolo'

def parse_list(value):
    try:
        return ast.literal_eval(value)
    except ValueError:
        return value

def parse_float(value):
    try:
        return float(ast.literal_eval(value)[0])
    except (ValueError, SyntaxError, TypeError):
        try:
            return float(value)
        except ValueError:
            return value

# Create a Path object for the directory
directory = Path(directory_path)

# List all directories in the specified directory
directories = [item for item in directory.iterdir() if item.is_dir()]

for dir in directories:
    if "run_20250620131410" in str(dir):
        # Read the dfagent.csv data
        df = pd.read_csv(
            str(dir) + '/dfagent.csv',
            converters={
                'epsilon_curve': parse_list,
                'action': parse_list,
                'latencies': parse_float,
                'reward': parse_float,
                'deviations': parse_float
            }
        )

        # Read hyperparameters
        with open(str(dir) + '/hyperparameters.txt', 'r') as file:
            data = file.read().replace('\n', ' ')
            print(data)
        
        # Extract the number from the matched group
        pattern = re.compile(r'no_of_episodes: (\d+) len_of_episode: (\d+)')
        match = pattern.search(data)
        
        if match:
            number1 = int(match.group(1))
            number2 = int(match.group(2))
            print("Number 1:", number1)
            print("Number 2:", number2)

            d = 50
            x_axis = range(1, int(int(number1) / d))  # Exclude zero from x-axis
            # Define the number of data points for each group
            N = int(number2) * d

            # Clip the values in the DataFrame
            df['latencies'] = df['latencies'].clip(lower=0, upper=20)

            # Group by N data points and calculate the mean
            columns_to_exclude = ['action', 'load']
            columns_to_group = [col for col in df.columns if col not in columns_to_exclude]

            # Group by specified interval excluding the last two columns
            grouped = df[columns_to_group].groupby(df.index // N)
            result = grouped.mean()

            # Transform the reward for better visualization
            # ep_rew = (np.abs(result['reward']))[:-1] 
            ep_lat = result['latencies'][:-1]
            ep_dev = result['deviations'][:-1]
            ep_a_r = (result['episode_access_rate'] * 100)[:-1]

            SMALL_SIZE = 26
            MEDIUM_SIZE = 24
            BIGGER_SIZE = 18

            plt.rc('font', size=SMALL_SIZE) # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE) # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

            # Define x-axis ticks
            x_ticks = np.arange(10, number1 // d + 1, 10)

            # Common function to format y-tick labels
            def format_y_ticks(ax, format_str='{:.2f}'):
                yticks = ax.get_yticks()
                ax.yaxis.set_major_locator(FixedLocator(yticks))
                ax.yaxis.set_major_formatter(FixedFormatter([format_str.format(ytick) for ytick in yticks]))

            # Helper function to determine the appropriate y-tick format
            def get_format_string(values):
                if np.max(values) > 1:
                    return '{:.1f}'
                else:
                    return '{:.2f}'

            # # Plot the reward graph
            # plt.figure(figsize=(8, 7))
            # plt.plot(x_axis, ep_rew, color='red', marker='o', markerfacecolor='black')
            # plt.xlabel('Average Episode Number', fontsize=24)
            # plt.ylabel('Reward', fontsize=24)
            # plt.grid(True, linestyle='--', alpha=0.7)
            # plt.xlim(left=1)
            # plt.xticks(x_ticks)
            # format_y_ticks(plt.gca(), get_format_string(ep_rew))
            # plt.savefig(str(dir) + "/episodic_rew.pdf", bbox_inches='tight')
            # plt.close()

            # Plot the deviations graph
            plt.figure(figsize=(8, 7))
            plt.plot(x_axis, ep_dev, color='red', marker='o', markerfacecolor='black')
            plt.xlabel('Average Episode Number', fontsize=24)
            plt.ylabel('Deviations (seconds)', fontsize=24)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(left=1)
            plt.xticks(x_ticks)
            format_y_ticks(plt.gca(), get_format_string(ep_dev))
            plt.savefig(str(dir) + "/episodic_dev.pdf", bbox_inches='tight')
            plt.close()

            # Plot the latencies graph
            plt.figure(figsize=(8, 7))
            plt.plot(x_axis, ep_lat, color='red', marker='o', markerfacecolor='black')
            plt.xlabel('Average Episode Number', fontsize=24)
            plt.ylabel('Latencies (seconds)', fontsize=24)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(left=1)
            plt.xticks(x_ticks)
            format_y_ticks(plt.gca(), get_format_string(ep_lat))
            plt.savefig(str(dir) + "/episodic_lat.pdf", bbox_inches='tight')
            plt.close()

            # # Plot the episode access rate graph
            # plt.figure(figsize=(8, 7))
            # plt.plot(x_axis, ep_a_r, color='red', marker='o', markerfacecolor='black')
            # plt.xlabel('Average Episode Number', fontsize=24)
            # plt.ylabel('Access Rate (%)', fontsize=26)
            # plt.grid(True, linestyle='--', alpha=0.7)
            # plt.xlim(left=1)
            # plt.xticks(x_ticks)
            # format_y_ticks(plt.gca(), get_format_string(ep_a_r))
            # plt.savefig(str(dir) + "/episodic_a_r.pdf", bbox_inches='tight')
            # plt.close()

            # Plot the episode access rate graph with values on points
            # Plot the episode access rate graph with values on points
            plt.figure(figsize=(8, 7))
            plt.plot(x_axis, ep_a_r, color='red', marker='o', markerfacecolor='black')

            ep_a_r = (result['episode_access_rate'] * 100)[:-1]
            print("Last Access Rate Value:", ep_a_r.iloc[-1])


            # Annotate each point with its exact value
            for i, txt in enumerate(ep_a_r):
                plt.text(x_axis[i], ep_a_r[i], f"{txt:.1f}", fontsize=14, ha='right', va='bottom', color='black')

            plt.xlabel('Average Episode Number', fontsize=24)
            plt.ylabel('Access Rate (%)', fontsize=26)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(left=1)
            plt.xticks(x_ticks)
            format_y_ticks(plt.gca(), get_format_string(ep_a_r))
            plt.savefig(str(dir) + "/episodic_a_r.pdf", bbox_inches='tight')
            plt.close()


        else:
            print("Pattern not found in the directory path.")


    else:
        print("Pattern not found in the directory path.")
