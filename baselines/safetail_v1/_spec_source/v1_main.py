import time as ti
import random
from time import sleep
import pandas as pd
from agent import DQNAgent
from agent import *
import pickle
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yolo.constants_yolo as yolo_constants
import noise.constants_noise as noise_constants
import instance.constants_instance as instance_constants
import argparse
import datetime

import warnings
warnings.filterwarnings('ignore')
# if tf.test.is_gpu_available():
#     # Print the GPU device that TensorFlow is using
#     print("TensorFlow is using GPU:", tf.test.gpu_device_name())
# else:
#     print("No GPU found; TensorFlow is using CPU.")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def normalize(df , a , b):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('data.csv')
    # Extract the timestamp column
    timestamps = df['Timestamp']
    # Extract the user columns for normalization
    user_columns = df.drop(columns=['Timestamp'])
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(a, b))
    # Normalize the user columns
    normalized_users = scaler.fit_transform(user_columns)
    # Round the normalized values to the nearest integer
    normalized_users = normalized_users.round().astype(int)
    # Create a new DataFrame with normalized integer values
    df_normalized = pd.DataFrame(normalized_users, columns=user_columns.columns)
    # Add the timestamp column back to the DataFrame
    df_normalized['Timestamp'] = timestamps
    # Reorder columns to have timestamp as the first column
    df_normalized = df_normalized[['Timestamp'] + user_columns.columns.tolist()]
    return df_normalized

# Use the constant value from constants.py

def get_load_at_time(time , df , beta):
    return df.iloc[time, 1:beta+1].values

def get_random_resolution(resolutions):
    res = random.choice(resolutions)
    return int(res.split('x')[0]) * int(res.split('x')[1])


def get_random_file_size(task ,yolo_file_size_df , instance_file_size_df,noise_df):
    #convert everything to KB
    if task == 'yolo':
        return yolo_file_size_df['col1'].sample().values[0]
    if task == 'instance':
        return instance_file_size_df['Size (bytes)'].sample().values[0]/1024
    if task == 'noise':
        return noise_df['file_size'].sample().values[0]/1024


def my_function(task):
    os.makedirs(task, exist_ok=True)
    new_directory = task + "/run_" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
   
    if task == 'yolo':
        no_of_episodes = yolo_constants.no_of_episodes
        len_of_episode = yolo_constants.len_of_episode
        nS = yolo_constants.nS
        nA = yolo_constants.nA
        learning_rate = yolo_constants.learning_rate
        discount_rate = yolo_constants.discount_rate
        gamma_decay = yolo_constants.gamma_decay
        batch_size = yolo_constants.batch_size
        max_bandwidth = yolo_constants.max_bandwidth
        max_load = yolo_constants.max_load
        median_computation_delay = yolo_constants.median_computation_delay
        alpha = yolo_constants.alpha
        beta = yolo_constants.beta
        epochs = yolo_constants.epochs
        

    if task == 'noise':
        no_of_episodes = noise_constants.no_of_episodes
        len_of_episode = noise_constants.len_of_episode
        nS = noise_constants.nS
        nA = noise_constants.nA
        learning_rate = noise_constants.learning_rate
        discount_rate = noise_constants.discount_rate
        gamma_decay = noise_constants.gamma_decay
        batch_size = noise_constants.batch_size
        max_bandwidth = noise_constants.max_bandwidth
        max_load = noise_constants.max_load
        median_computation_delay = noise_constants.median_computation_delay
        alpha = noise_constants.alpha
        beta = noise_constants.beta
        epochs = noise_constants.epochs
        
    if task == 'instance':
        no_of_episodes = instance_constants.no_of_episodes
        len_of_episode = instance_constants.len_of_episode
        nS = instance_constants.nS
        nA = instance_constants.nA
        learning_rate = instance_constants.learning_rate
        discount_rate = instance_constants.discount_rate
        gamma_decay = instance_constants.gamma_decay
        batch_size = instance_constants.batch_size
        max_bandwidth = instance_constants.max_bandwidth
        max_load = instance_constants.max_load
        median_computation_delay = instance_constants.median_computation_delay
        alpha = instance_constants.alpha
        beta = instance_constants.beta
        epochs = instance_constants.epochs
    
    df = pd.read_csv('data.csv', index_col=False)
    yolo_file_size_df = pd.read_csv('filesize.csv', index_col=False)
    instance_file_size_df = pd.read_csv('instance/image_file_sizes.csv', index_col=False)
    noise_df = pd.read_csv('noise/noise_final_data_19May.csv', index_col=False)
    resolution_df  = pd.read_csv('inst2.csv', header=None, names=range(12))
    propogation_delay_data = pickle.load(open('btp/ping_data.pkl', 'rb'))

    resolutions=[]
    
    for row_no in range(len(resolution_df)):
        resolutions.append(resolution_df.iloc[row_no,0].split()[3])

    df = normalize(df, 1, max_load)

    agent = DQNAgent(states = nS, actions = nA, alpha = alpha, reward_gamma = discount_rate, epsilon = 1.0, epsilon_min = 0.000001, epsilon_decay = gamma_decay, batch_size = batch_size, beta = beta, median_computation_delay = median_computation_delay, learning_rate = learning_rate , task = task , epochs = epochs)
    state = {}
    next_state = {}
    os.makedirs(new_directory, exist_ok=True)
    for e in range(no_of_episodes): # iterate over new episodes of the game
        start_time = ti.time()

        if task == 'yolo':
            state={'LOAD':[] , 'RESOLUTION':[], 'BANDWIDTH':[] , 'MESSAGE_SIZE':[],'PROPOGATION':[]} 
            next_state={'LOAD':[] , 'RESOLUTION':[], 'BANDWIDTH':[] , 'MESSAGE_SIZE':[],'PROPOGATION':[]}

        if task == 'noise':
            state={'LOAD':[] , 'BANDWIDTH':[], 'MESSAGE_SIZE':[],'PROPOGATION':[]}
            next_state={'LOAD':[] , 'BANDWIDTH':[], 'MESSAGE_SIZE':[],'PROPOGATION':[]}
        
        if task == 'instance':
            # state={'LOAD':{} ,'MESSAGE_SIZE':{} , 'BANDWIDTH':{} }
            # new_state={'LOAD':{} , 'MESSAGE_SIZE':{} , "BANDWIDTH":{} }
            state={'LOAD':[] ,'MESSAGE_SIZE':[]  , 'BANDWIDTH':[] , 'PROPOGATION':[] }
            next_state={'LOAD':[] ,'MESSAGE_SIZE':[]  , 'BANDWIDTH':[] , 'PROPOGATION':[] }

        # print("Stae: ", state)


        start_row = random.randint(0,len(df))   # starting point of row in edge_vicity_file then increament one for each time in a round robin manner   
        state["LOAD"] = get_load_at_time((start_row)%len(df) , df , beta)
        # random.shuffle(state["LOAD"])
        if "RESOLUTION" in state.keys():
            state['RESOLUTION'] = get_random_resolution(resolutions)
        if "MESSAGE_SIZE" in state.keys():
            state['MESSAGE_SIZE'] = get_random_file_size(args.task , yolo_file_size_df , instance_file_size_df,noise_df)
        if  "BANDWIDTH" in state.keys():
            state['BANDWIDTH'] = max_bandwidth
        if "PROPOGATION" in state.keys():
            state['PROPOGATION'] = [random.choice(propogation_delay_data[i-1]) for i in state['LOAD']]
        
        # print(state)
        for time in range(len_of_episode): 

            agent.load_arr.append(state['LOAD'])
            next_state['LOAD'] = get_load_at_time((start_row+time+1)%len(df) , df , beta)
            # random.shuffle(next_state['LOAD'])
            if "RESOLUTION" in state.keys():
                next_state['RESOLUTION'] = get_random_resolution(resolutions)
            if "MESSAGE_SIZE" in state.keys():
                next_state['MESSAGE_SIZE'] = get_random_file_size(args.task , yolo_file_size_df , instance_file_size_df,noise_df)
            if "BANDWIDTH" in state.keys():
                next_state['BANDWIDTH'] = max_bandwidth
            if "PROPOGATION" in state.keys():
                next_state['PROPOGATION'] = [random.choice(propogation_delay_data[i-1]) for i in next_state['LOAD']]
            # print("next_state: ", next_state)

            action_arr , action = agent.get_action(state) # action is a number between 1  and 2^beta

            reward  = agent.reward(action_arr,state)
            # print("reward: ", reward)
            agent.store(state,action,reward,next_state) # remember the previous timestep's state, actions, reward, etc.        
            agent.epsilon_curve.append(agent.epsilon)
            state = next_state.copy()

        end_time = ti.time()

        if e % 100 == 0:
            keras.backend.clear_session()
        
        if e % 1000 == 0:
            agent.model.save(new_directory+"/model_"+ str(e) + ".keras")
        

        print("Time taken for episode: ", end_time - start_time)
        print("episode: {}/{}" # print the episode's score and agent's epsilon
                .format(e, no_of_episodes))


    # LOGGING AND SAVING

    
    agent.model.save(new_directory+"/model_"+ str(no_of_episodes) + ".keras")

    # write all the hyperparameters to a file
    with open(new_directory+"/hyperparameters.txt", "w") as f:
        f.write("task: {}\n".format(task))
        f.write("no_of_episodes: {}\n".format(no_of_episodes))
        f.write("len_of_episode: {}\n".format(len_of_episode))
        f.write("alpha: {}\n".format(alpha))
        f.write("learning_rate: {}\n".format(learning_rate))
        f.write("batch_size: {}\n".format(batch_size))
        f.write("gamma_decay: {}\n".format(gamma_decay))
        f.write("beta: {}\n".format(beta))
        f.write("max_load: {}\n".format(max_load))
        f.write("max_bandwidth: {}\n".format(max_bandwidth))
        f.write("median_computation_delay: {}\n".format(median_computation_delay))
        f.write("epochs: {}\n".format(epochs))

        
        

    data_loss = {
        'loss': agent.loss,
        'val_loss': agent.val_loss,
    }


    data_agent = {
        'reward': agent.rewards,
        'latencies': agent.latencies,
        'deviations': agent.deviations,
        'episode_access_rate':agent.episode_access_rate,
        'epsilon_curve' : agent.epsilon_curve,
        'action':agent.action,
        'load':agent.load_arr
    }

    print("len of rewards: ", len(agent.rewards))
    print("len of latencies: ", len(agent.latencies))
    print("len of deviations: ", len(agent.deviations))
    print("len of episode_access_rate: ", len(agent.episode_access_rate))
    print("len of epsilon_curve: ", len(agent.epsilon_curve))
    print("len of action: ", len(agent.action))
    print("len of load: ", len(agent.load_arr))



    df_loss = pd.DataFrame(data_loss)
    df_agent = pd.DataFrame(data_agent)

    df_loss.to_csv(new_directory+"/dfloss.csv", index=False)
    df_agent.to_csv(new_directory+"/dfagent.csv", index=False)

    plt.plot(range(len(agent.loss)), agent.loss)
    plt.savefig(new_directory + "/loss.png")
    plt.close()

    plt.plot( range(len(agent.val_loss)), agent.val_loss)
    plt.savefig(new_directory + "/val_loss.png")
    plt.close()

    plt.plot( range(len(agent.rewards)), agent.rewards)
    plt.savefig(new_directory + "/rewards.png")
    plt.close()


    plt.plot( range(len(agent.latencies)), agent.latencies)
    plt.savefig(new_directory + "/latencies.png")
    plt.close()

    plt.plot( range(len(agent.deviations)), agent.deviations)
    plt.savefig(new_directory + "/deviations.png")
    plt.close()

    plt.plot( range(len(agent.epsilon_curve)), agent.epsilon_curve)
    plt.savefig(new_directory + "/epsilon_curve.png")
    plt.close()

    plt.plot( range(len(agent.episode_access_rate)), agent.episode_access_rate)
    plt.savefig(new_directory + "/episodic_access_rate.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enter the safety critical task')
    parser.add_argument('task', help='Enter the tasks you want the DQN agent to perform')
    args = parser.parse_args()

    if args.task == 'yolo':
        my_function('yolo')
    if args.task == 'noise':
        my_function('noise')
    if args.task == 'instance':
        my_function('instance')
    else:
        print("Invalid task")

