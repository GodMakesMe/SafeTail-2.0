import time
import random

from time import sleep
import pandas as pd
import constants
from agent import *
import pickle
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras 
import numpy as np
import argparse
import re



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
warnings.filterwarnings('ignore')
with open('/home/aman/ut_dqn/instance/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
with open('/home/aman/ut_dqn/instance/regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)

mlp_regr_model = open('/home/aman/yolo_thread_mlp_regressor_model.pkl', 'rb')
mlp_regressor = pickle.load(mlp_regr_model)
with open('/home/aman/yolo_thread_scaler.pkl', 'rb') as file:
        mlp_scaler = pickle.load(file)

def normalize(df , a , b):
    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        column_normalized = a + (df[column] - min_val) * (b-a) / (max_val - min_val)
        df[column] = column_normalized
    df = df.round().astype(int)
    return df

def get_computation_delay(state,node,task):
    if task == "yolo":
        number_of_users = state['LOAD'][node]
        if "RESOLUTION" not in state.keys():
            resolution = get_random_resolution(resolutions)
        else:
            resolution = state['RESOLUTION']
        


        inp = np.array([number_of_users, resolution])
        X_test = np.array([[inp[0],inp[1]]])
        X_test = np.array(X_test)
        # print(X_test)
        st_dev = [2.7965017993200005, 6.393525858045778, 5.515155335473335, 7.100407896557774, 6.909932092510028, 7.124802705029803, 7.972889920066626, 9.872241111272556, 8.936860060306417, 12.450634470736823, 12.444299205142089, 11.830614817831743, 14.73346686870066, 17.763701566033472, 14.910366835514143, 19.22145836272576, 17.4644685302588, 19.87528975084137, 23.439168758290045, 20.457122880591005]
        pred = mlp_regressor.predict(mlp_scaler.transform(X_test))[0]
        return (pred + np.random.normal(0 , st_dev[int(inp[0])-1] ,1))/1000
        # return mlp_regressor.predict(np.array([number_of_users, resolution]))
    
    if task == "noise":
        return random.choice(latencies[state['LOAD'][node]-1])
    
    if task == "instance":
        number_of_users = state['LOAD'][node]
        size = state['MESSAGE_SIZE']
        X = np.array([[size, number_of_users]])
        X = scaler.transform(X)
        # std = [0.5375752895974742, 0.863496593667569, 0.8602990093104789, 0.8389007123017749, 0.746915476099896]
        std = [4.182686523232169, 27.4826173554251, 35.12740655487459, 39.941942069458854, 40.001337579530286]
        predict = regressor.predict(X)
        noise = np.random.normal(0, std[number_of_users-1], len(predict))
        predict_new = predict + abs(noise)/2
        return predict_new

    
def get_random_latency(state , num,task):
  subsets = []
  total_subsets = get_subsets(set([x for x in range(constants.beta)]))
  for set_s in total_subsets:
    if len(set_s) == num :
        subsets.append(set_s)
  return_arr = random.choice(subsets)

  obs_latency = 100000000
  for node in return_arr :
    computaion_delay_node = get_computation_delay(state,node,task)
    tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'] / state['LOAD'][node]  ,state['BANDWIDTH'] / state['LOAD'][node] )
    # tramission_delay_node =0
    propogation_delay_node =  get_propogation_delay(state['LOAD'][node])
    node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node
    obs_latency = min(obs_latency , node_latency)

  return obs_latency

def get_agent_latency(agent,state,task,beta):
  # print(state)
  subsets = get_subsets(set([x for x in range(beta)]))
  state_flattened = get_state_input(state)
  print(state_flattened)
  if task == "yolo":
    state_flattened = np.array(state_flattened).reshape(1,13)
  if task == "instance":
    state_flattened = np.array(state_flattened).reshape(1,12)
  action_vals = agent.predict(state_flattened) #Exploit: Use the NN to predict the correct action from this state
  # print(state_flattened)
  # print(action_vals)
  action = np.argmax(action_vals[0])

  print(subsets[action])

  return_arr = subsets[action]

  obs_latency = 100000000
  for node in return_arr :
    computaion_delay_node = get_computation_delay(state,node,task)
    tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'] / state['LOAD'][node]  ,state['BANDWIDTH'] / state['LOAD'][node] )

    # tramission_delay_node = 0
    propogation_delay_node =  get_propogation_delay(state['LOAD'][node])
    node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node
    obs_latency = min(obs_latency , node_latency)
  # print(subsets[action])
  return obs_latency , len(subsets[action])/beta 

def get_minimum_latency(state,task,beta):
  subsets = get_subsets(set([x for x in range(beta)]))
  return_arr = subsets[-1]  # last subset has the whole sensors
  obs_latency = 100000000
  for node in return_arr :
    computaion_delay_node = get_computation_delay(state,node,task)

    tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'] / state['LOAD'][node]  ,state['BANDWIDTH'] / state['LOAD'][node] )

    # tramission_delay_node =0
    propogation_delay_node =  get_propogation_delay(state['LOAD'][node])
    node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node
    obs_latency = min(obs_latency , node_latency)
  return obs_latency

import random

def get_min_load_latency(state, task, num_nodes=3):
    # Randomly select three nodes out of five
    selected_nodes = random.sample(range(len(state["LOAD"])), num_nodes)
    
    # Choose the node with the minimum load among the selected nodes
    min_load_node = min(selected_nodes, key=lambda i: state["LOAD"][i])
    
    # Calculate latency for the selected node
    computaion_delay_node = get_computation_delay(state, min_load_node, task)
    tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'], state['BANDWIDTH'] / state['LOAD'][min_load_node], state['BANDWIDTH'] / state['LOAD'][min_load_node])
    propogation_delay_node = get_propogation_delay(state['LOAD'][min_load_node])
    node_latency = computaion_delay_node + tramission_delay_node + propogation_delay_node
    
    return node_latency



def get_min_propogation_latency(state,task):

  node = 0
  for i in range(len(state["PROPOGATION"])):
    if state['PROPOGATION'][i] < state['PROPOGATION'][node]:
      node = i
  
  computaion_delay_node = get_computation_delay(state,node,task)
  tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'] / state['LOAD'][node]  ,state['BANDWIDTH'] / state['LOAD'][node] )
  # tramission_delay_node = 0
  propogation_delay_node =  get_propogation_delay(state['LOAD'][node])
  node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node
  return node_latency

  



  # pair_array = [[i,state['LOAD'][i]] for i in return_arr]
  # pair_array.sort(key = lambda x: x[1])
  # print(pair_array)

  # obs_latency = 100000000
  # for pair in pair_array:
  #   node = pair[0]
  #   computaion_delay_node = get_computation_delay(state,node,task)
  #   tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'][node],state['BANDWIDTH'][node] )
  # # tramission_delay_node = 0
  #   propogation_delay_node =  get_propogation_delay()
  #   node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node
  #   obs_latency = min(obs_latency , node_latency)
  # return obs_latency

def get_random_file_size(task ):
    with open('filesize.csv', 'rb') as f:
        yolo_file_size_df = pd.read_csv(f)
    with open('instance/image_file_sizes.csv', 'rb') as f:
        instance_file_size_df = pd.read_csv(f)

    if task == 'yolo':
        return yolo_file_size_df['col1'].sample().values[0]
    if task == 'instance':
        return instance_file_size_df['Size (bytes)'].sample().values[0]/1024


def get_min_transmission_delay(state,task):
  # node = 1

  # node = state['BANDWIDTH'].index(max(state['BANDWIDTH']))
  # print(node)
  # print(state['BANDWIDTH'])
  # computaion_delay_node = get_computation_delay(state,node,task)
  # # tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'][node]  ,state['BANDWIDTH'][node] )
  # tramission_delay_node = 0
  # propogation_delay_node =  0
  # node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node
  # return node_latency
  return 0

def load_model(model_path):
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Provide the path to the run folder")
    parser.add_argument("run_config", help="Path to the run config folder")
    args = parser.parse_args()
    # Load the model
    if not os.path.exists(args.run_config):
        print("Model path does not exist!")
        return
    else:
        print("Model path exists!")
        path = re.compile(r"model_(\d+).keras")
        model_files = [file for file in os.listdir(args.run_config) if path.match(file)]  
        if not model_files:
            print("No model files found in the folder!")
            return
        model_files.sort(key=lambda x: int(path.match(x).group(1)))
        loaded_model = load_model(args.run_config + '/' + model_files[-1])

        if loaded_model:
            print("Model loaded successfully!")
            df = pd.read_csv('data.csv', index_col=False)
            resolution_df  = pd.read_csv('inst2.csv', header=None, names=range(12))
            resolutions=[]
            for row_no in range(len(resolution_df)):
                resolutions.append(resolution_df.iloc[row_no,0].split()[3])
            yolo_file_size_df = pd.read_csv('filesize.csv', index_col=False)
            propogation_delay_data = pickle.load(open('btp/ping_data.pkl', 'rb'))

            # read hyperparameters
            with open(args.run_config + '/hyperparameters.txt', 'rb') as f:
                hyperparameters = f.readlines()

            # read max_load and convert to int
            task = hyperparameters[0].decode('utf-8').split()[-1]
            max_bandwidth = int(hyperparameters[9].decode('utf-8').split()[-1])
            max_load = int(hyperparameters[8].decode('utf-8').split()[-1])
            beta= int(hyperparameters[7].decode('utf-8').split()[-1])
            df = normalize(df,1,max_load)
            
            agent = loaded_model
            agent_latencies = []
            global_min_delay_latencies = []
            min_load_latencies = []
            min_propagation_latencies = []
            random_latencies_1 = []
            random_latencies_2 = []
            random_latencies_3 = []
            random_latencies_4 = []
            access_rate = []

            N = 500

            for run in range(5):
              for index in df[0:1400].index:
                  if task == 'yolo':
                    state={'LOAD':{} , 'RESOLUTION':{}, 'BANDWIDTH':{} , 'MESSAGE_SIZE':{},'PROPOGATION':{}} 

                  if task == 'noise':
                    state={'LOAD':{}}
                  
                  if task == 'instance':
                    state={'LOAD':{} ,'MESSAGE_SIZE':{}  , 'BANDWIDTH':{} , 'PROPOGATION':{} }

                  resolution_tuple = random.choice(resolutions)
                  resolution = int(resolution_tuple.split('x')[0]) * int(resolution_tuple.split('x')[1])

                  if task == 'yolo':
                    state['RESOLUTION'] = resolution

                  current_load = []
                  for edge_device_no in range(beta):
                      current_load.append(df.iloc[ index , 1+ edge_device_no])
                  
                    
                  state['LOAD'] = current_load
                  state['MESSAGE_SIZE'] = get_random_file_size(task)
                  state['BANDWIDTH'] =  max_bandwidth
                  state['PROPOGATION'] = [random.choice(propogation_delay_data[i-1]) for i in state['LOAD']]
                  # print(state)

                  min_latency = get_minimum_latency(state,task,beta)
                  min_load_latency = get_min_load_latency(state,task)
                  agent_latency  , gamma = get_agent_latency(agent,state,task,beta)
                  random_latency_1 = get_random_latency(state,1,task)
                  random_latency_2 = get_random_latency(state,2,task)
                  random_latency_3 = get_random_latency(state,3,task)
                  random_latency_4 = get_random_latency(state,4,task)
                  min_propogation_delay = get_min_propogation_latency(state,task)


                  agent_latencies.append(agent_latency)
                  global_min_delay_latencies.append(min_latency)
                  min_load_latencies.append(min_load_latency)
                  random_latencies_1.append(random_latency_1)
                  random_latencies_2.append(random_latency_2)
                  random_latencies_3.append(random_latency_3)
                  random_latencies_4.append(random_latency_4)
                  min_propagation_latencies.append(min_propogation_delay)
                  access_rate.append(gamma)
            

            data = {'MIN': global_min_delay_latencies, 'DQN': agent_latencies, 'RAND_1': random_latencies_1, 'RAND_2': random_latencies_2 , 'RAND_3': random_latencies_3 , 'RAND_4': random_latencies_4, 'MIN_LOAD': min_load_latencies, 'MIN_PROP': min_propagation_latencies}
            df = pd.DataFrame(data)
            df.to_csv( args.run_config+'/test_latency.csv', index=False)

            for i in [50,80,90,95,99]:


              min_latency_percentile = np.percentile(global_min_delay_latencies, i)
              agent_latency_percentile = np.percentile(agent_latencies, i)
              min_propagation_latency_percentile = np.percentile(min_propagation_latencies, i)
              min_load_latency_percentile = np.percentile(min_load_latencies, i)
              random_latency_1_percentile = np.percentile(random_latencies_1, i)
              random_latency_2_percentile = np.percentile(random_latencies_2, i)
              random_latency_3_percentile = np.percentile(random_latencies_3, i)
              random_latency_4_percentile = np.percentile(random_latencies_4, i)
              

              dic = {
                    'min': min_latency_percentile,
                    'dqn': agent_latency_percentile,
                    'random_1': random_latency_1_percentile,
                    'random_2': random_latency_2_percentile,
                    'random_3': random_latency_3_percentile,
                    'random_4': random_latency_4_percentile,
                    'min_load': min_load_latency_percentile,
                    'min_prop': min_propagation_latency_percentile
                    
                }

              plt.rcParams['figure.figsize'] = 12, 7.5
              SMALL_SIZE = 15
              MEDIUM_SIZE = 18
              BIGGER_SIZE = 18
              plt.rc('font', size=SMALL_SIZE) # controls default text sizes
              plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
              plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
              plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
              plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
              plt.rc('legend', fontsize=SMALL_SIZE) # legend fontsize
              plt.rc('figure', titlesize=BIGGER_SIZE)

              plt.bar(dic.keys(), dic.values(), color=['blue', 'orange', 'green', 'red', 'purple', 'brown' , 'pink' , 'black']) 
              plt.savefig(args.run_config + f"/percentile_{i}.png", bbox_inches='tight')
              plt.close()

              # plot access rate
            x_axis = range(len(agent_latencies) // N)
            access_rate_all = [np.mean(access_rate[i:i+N]) for i in range(0,len(access_rate),N)]
            print(access_rate_all)
            plt.plot(x_axis, access_rate_all, label='DQN', color='orange', marker='o')
            plt.xlabel("Average Episodes")
            plt.ylabel("Access Rate")
            plt.title("Access rate in Testing")
            plt.legend()
            plt.savefig(args.run_config + "/access_rate.png", bbox_inches='tight')

              













#               plt.rcParams['figure.figsize'] = 12, 7.5
#               SMALL_SIZE = 15
#               MEDIUM_SIZE = 18
#               BIGGER_SIZE = 18
#               plt.rc('font', size=SMALL_SIZE) # controls default text sizes
#               plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
#               plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
#               plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
#               plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
#               plt.rc('legend', fontsize=SMALL_SIZE) # legend fontsize
#               plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
#               plt.rc('axes', axisbelow=True) # fontsize of the figure title


#               patterns = ["\\\\\\", "//", "xxx", "+++", 'ooo', '***', '...', '---']
#               colors = ['dodgerblue', "hotpink", "green", 'dimgrey', "orange", "red", "blue", "black", "purple"]
#               bar_width = 0.2

#               # plt.bar(list(dic.keys()), list(dic.values()), width=bar_width, color=['blue', 'orange', 'green', 'red', 'purple', 'brown' , 'pink' , 'black'],edgecolor = colors[0], hatch=patterns[0])
#               # index = np.arange(4)
#               print("random_1",dic['random_1'])

#               plt.bar(np.arange(3),[dic['min_comp_1'],dic['min_comp_2'],dic['min_comp_3']], width=bar_width,edgecolor = colors[0], hatch=patterns[0], color='white', label='min_comp')
#               plt.bar(np.arange(3) + bar_width,[dic['random_1'],dic['random_2'],dic['random_3']], width=bar_width,edgecolor = colors[1], hatch=patterns[1], color='white', label='rand')
#               plt.bar(1 + 2*bar_width,[dic['dqn']], width=bar_width,edgecolor = colors[2], hatch=patterns[3], color='white', label='dqn')
#               plt.bar(2+ 3*bar_width,[dic['min']], width=bar_width,edgecolor = colors[4], hatch=patterns[4], color='white', label='min')



#               plt.xlabel("Access Rate")
#               plt.ylabel("Latency (in sec)")
#               plt.title(' ')

#               plt.xticks([0.1,1.2,2.1,2.6], [0.25, 0.5, 0.75 , 1])
#               # plt.xtick()
#               plt.legend(loc='upper right',ncol=2)

#               plt.grid(True, linestyle='--', alpha=0.7)

#               plt.savefig(args.run_config + f"/percentile_{i}.png", bbox_inches='tight')
#               plt.close()

#             x_axis = range(len(agent_latencies) // N)
#             # temp_latencies1 = [np.mean(global_min_delay_latencies[i:i+N]) for i in range(0,len(global_min_delay_latencies),N)]
#             # temp_latencies2 = [np.mean(agent_latencies[i:i+N]) for i in range(0,len(agent_latencies),N)]
#             # temp_latencies3 = [np.mean(random_latencies_1[i:i+N]) for i in range(0,len(random_latencies_1),N)]
#             # temp_latencies4 = [np.mean(random_latencies_2[i:i+N]) for i in range(0,len(random_latencies_2),N)]
#             # temp_latencies5 = [np.mean(random_latencies_3[i:i+N]) for i in range(0,len(random_latencies_3),N)]
#             # temp_latencies6 = [np.mean(random_latencies_4[i:i+N]) for i in range(0,len(random_latencies_4),N)]
#             # temp_latencies7 = [np.mean(min_computation_delay_latencies[i:i+N]) for i in range(0,len(min_computation_delay_latencies),N)]
#             # # temp_latencies8 = [np.mean(min_transmission_delay_latencies[i:i+N]) for i in range(0,len(min_transmission_delay_latencies),N)]
#             access_rate_all = [np.mean(access_rate[i:i+N]) for i in range(0,len(access_rate),N)]


#             plt.plot(x_axis, access_rate_all, label='DQN', color='orange', marker='o')
#             plt.xlabel("Average Episodes")
#             plt.ylabel("Access Rate")
#             plt.title("Access rate in Testing")
#             plt.legend()
#             plt.savefig(args.run_config + "/access_rate.png", bbox_inches='tight')
#             plt.close()


if __name__ == "__main__":
    main()


