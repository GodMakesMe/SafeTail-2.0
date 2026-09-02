import random
import numpy as np
from collections import deque
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential

from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd

import os # for creating directories
import pickle
import math
import time
from mlp_regressor import predict

# Read the pickle file latenices.pkl
with open('/home/iiitd/Desktop/aman/ut_dqn/noise/latencies.pkl', 'rb') as f:
    latencies = pickle.load(f)

with open('/home/iiitd/Desktop/aman/ut_dqn/instance/scaler.pkl', 'rb') as f:
    instance_scaler = pickle.load(f)
with open('/home/iiitd/Desktop/aman/ut_dqn/instance/regressor.pkl', 'rb') as f:
    instance_regressor = pickle.load(f)

with open('/home/iiitd/Desktop/aman/ut_dqn/noise/scaler.pkl', 'rb') as f:
    noise_scaler = pickle.load(f)

with open('/home/iiitd/Desktop/aman/ut_dqn/noise/mlp_regressor.pkl', 'rb') as f:
    noise_regressor = pickle.load(f)


mlp_regr_model = open('/home/iiitd/Desktop/aman/yolo_thread_mlp_regressor_model.pkl', 'rb')
mlp_regressor = pickle.load(mlp_regr_model)
with open('/home/iiitd/Desktop/aman/yolo_thread_scaler.pkl', 'rb') as file:
        mlp_scaler = pickle.load(file)
resolution_df  = pd.read_csv('inst2.csv', header=None, names=range(12))
resolutions=[]
ping_data = pickle.load(open('btp/ping_data.pkl', 'rb'))


for row_no in range(len(resolution_df)):
    resolutions.append(resolution_df.iloc[row_no,0].split()[3])

def get_propogation_delay(num):
    return random.choice(ping_data[num-1])

def get_tramission_delay(message_size , uplink , downlink):
    message_size = 8 * message_size # Kb
    # uplink is in Mb/s
    # downlink is in Mb/s
    uplink = uplink * 1000 # Kb/s
    downlink = downlink * 1000 # Kb/s
    return (message_size / uplink) + (message_size / downlink)
    # return 0
def get_random_resolution(resolutions):
    res = random.choice(resolutions)
    return int(res.split('x')[0]) * int(res.split('x')[1])

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
        st_dev = [2.7965017993200005, 6.393525858045778, 5.515155335473335, 7.100407896557774, 6.909932092510028, 7.124802705029803, 7.972889920066626, 9.872241111272556, 8.936860060306417, 12.450634470736823, 12.444299205142089, 11.830614817831743, 14.73346686870066, 17.763701566033472, 14.910366835514143, 19.22145836272576, 17.4644685302588, 19.87528975084137, 23.439168758290045, 20.457122880591005]
        pred = mlp_regressor.predict(mlp_scaler.transform(X_test))[0]
        return (pred + np.random.normal(0 , st_dev[int(inp[0])-1] ,1))/1000
        # return mlp_regressor.predict(np.array([number_of_users, resolution]))
    
    if task == "noise":
        inp = np.array([state['LOAD'][node], state['MESSAGE_SIZE']])
        X = np.array([inp])
        X = noise_scaler.transform(X)
        st_dev = [0.025221,0.027035,0.027473,0.029890,0.031513]
        predict = noise_regressor.predict(X)
        noise = np.random.normal(0, st_dev[state['LOAD'][node]-1], len(predict))
        predict_new = predict + abs(noise)/2
        return predict_new
    
    if task == "instance":
        number_of_users = state['LOAD'][node]
        size = state['MESSAGE_SIZE']
        X = np.array([[size, number_of_users]])
        X = instance_scaler.transform(X)
        # std = [0.5375752895974742, 0.863496593667569, 0.8602990093104789, 0.8389007123017749, 0.746915476099896]
        std = [4.182686523232169, 27.4826173554251, 35.12740655487459, 39.941942069458854, 40.001337579530286]
        predict = instance_regressor.predict(X)
        noise = np.random.normal(0, std[number_of_users-1], len(predict))
        predict_new = predict + abs(noise)/2
        return predict_new

def get_subsets(fullset):
        listrep = list(fullset)
        subsets = []
        for i in range(2**len(listrep)):
            subset = []
            for k in range(len(listrep)):
             if i & 1<<k:
                subset.append(listrep[k])
            subsets.append(subset)
        return subsets[1:]

def get_state_input(state):
    output = []
    for i in state['LOAD']:
        output.append(i)
    if "MESSAGE_SIZE" in state.keys():
        output.append(state['MESSAGE_SIZE'])
    if "RESOLUTION" in state.keys():
        output.append(state['RESOLUTION'])
    if "BANDWIDTH" in state.keys():
        output.append(state['BANDWIDTH'])
    if "PROPOGATION" in state.keys():
        for i in state['PROPOGATION']:
            output.append(i)
    # print("State Input" , output)
    return output
    


class DQNAgent:
    def __init__(self, states, actions, alpha, reward_gamma, epsilon,epsilon_min, epsilon_decay , batch_size , beta , median_computation_delay , learning_rate, task,epochs):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.reward_gamma = reward_gamma
        self.beta = beta
        self.median_computation_delay = median_computation_delay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate 
        self.task = task
        self.model = self.build_model()
        self.epochs = epochs
        # print(self.model.summary())  # Print model summary here
        self.loss = []
        self.val_loss = []
        self.exploit_or_explore = []
        self.epsilon_curve=[]
        self.episode_access_rate = []
        self.latencies = []
        self.deviations = []
        self.rewards = []
        self.action = []
        self.load_arr = []

    def build_model(self) :
        model = keras.Sequential() 
        model.add(keras.layers.Dense(self.nS*2, input_dim=self.nS, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nS*4, activation='sigmoid')) 
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.nA, activation='softmax')) 

        model.compile(loss='categorical_crossentropy', 
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)) 
        return model
    
    def store(self, state, action, reward , next_state):
        self.memory.append( ( get_state_input(state), action, reward , get_state_input(next_state)) )
    
    def get_action(self,state):
        no_of_edges = self.beta
        subsets = get_subsets(set([x for x in range(self.beta)]))
        valid_subsets = get_subsets(set([x for x in range(no_of_edges)]))
        state_flattened = get_state_input(state)
        state_flattened = np.array(state_flattened).reshape(1,self.nS)
        if np.random.rand() <= self.epsilon:
            self.exploit_or_explore.append("explore")
            action = np.random.randint(0,self.nA)
        else:
            self.exploit_or_explore.append("exploit")
            start_time = time.time()
            action_vals = self.model.predict(state_flattened , verbose =0) #Exploit: Use the NN to predict the correct action from this state
            end_time = time.time()
            print("Time taken to Predict: {:.4f} seconds".format(end_time - start_time))  
            # keras.backend.clear_session() # memory leak fix
            action = np.argmax(action_vals[0])

        return_arr = subsets[action]
   
        self.episode_access_rate.append(float(len(return_arr))/no_of_edges)
        self.action.append(subsets[action])
        return return_arr , action 

    def experience_replay(self, batch_size):
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory
        states, actions, rewards, next_states = map(np.array, zip(*minibatch))
        states = np.array(states)
        current_q = self.model.predict(states, verbose =0)
        # keras.backend.clear_session() # memory leak fix
        start_time = time.time()

        next_q_values = self.model.predict(next_states,verbose =0)
        # keras.backend.clear_session() # memory leak fix
        end_time = time.time()
        print("Time taken to Train: {:.4f} seconds".format(end_time - start_time))


        targets = current_q.copy()
        # print("rewards" , rewards)
        targets[np.arange(batch_size), actions] = (rewards) + self.reward_gamma * np.amax(next_q_values, axis=1) 
        # print("targets" , targets)

        hist = self.model.fit(states, targets, epochs=self.epochs, verbose=0 , validation_split=0.2)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        loss_sum = hist.history['loss'][0]
        val_loss_sum = hist.history['val_loss'][0]
        

        self.loss.append(loss_sum)
        self.val_loss.append(val_loss_sum)
        end_time = time.time()

        #Reshape for Keras Fit
        #Decay Epsilon
       

    def reward(self, action , state):
        # print("State" , state)
        # print("Action" , action)
        MEDIAN_LATENCY = self.median_computation_delay

        obs_latency = 100000000
        # print("Action" , action)
        # print("State" , get_state_input(state))
        for node in action :
            computaion_delay_node = get_computation_delay(state,node,self.task)
            tramission_delay_node = get_tramission_delay(state['MESSAGE_SIZE'] , state['BANDWIDTH'] / state['LOAD'][node] ,state['BANDWIDTH']/ state['LOAD'][node])
            propogation_delay_node = get_propogation_delay(state['LOAD'][node])
            # print("Node" , node , "Computation Delay" , computaion_delay_node , "Tramission Delay" , tramission_delay_node, "Propogation Delay" , propogation_delay_node)
            # propogation_delay_node =  get_propogation_delay()
            node_latency =  computaion_delay_node + tramission_delay_node + propogation_delay_node

            if self.task == "instance":
                node_latency = math.log(1+node_latency)
                node_latency = np.array([node_latency])
            
            if self.task == "noise":
                node_latency = math.log(1+node_latency)
                node_latency = np.array([node_latency])

            
            # node_latency =  computaion_delay_node
            obs_latency = min(obs_latency , node_latency)


        if abs(obs_latency - MEDIAN_LATENCY) < 1000:
            # reward1 = self.alpha*(len(action))
            # reward2 = math.log(1+abs(obs_latency - MEDIAN_LATENCY))
            # reward = -1* (self.alpha*(len(action)) + math.log(1+abs(obs_latency - MEDIAN_LATENCY)) ) 
            lamda = (obs_latency - MEDIAN_LATENCY)
            reward = 0
            delta = 0
            gamma = len(action)-1 
            # print("gamma" , gamma)

            if lamda < 0:
                delta = (self.alpha * np.exp(-1 * lamda))

            else:
                delta = (self.alpha * np.exp(lamda))
            
            if lamda == 0:
                reward = 0
                
            elif lamda > 0 and self.beta - gamma == 1:

                reward = 0

            elif lamda > 0 and self.beta - gamma > 1 :
            
                reward = (-1 * np.exp(self.beta - gamma - 1) * delta)[0]
                

            elif lamda < 0:
                
                reward = (-1 * np.exp(gamma) * delta)[0]

            self.latencies.append(obs_latency)
            self.deviations.append(abs(obs_latency - MEDIAN_LATENCY))
            self.rewards.append(reward)    
            # print("Reward" , reward)
            # print("Latency" , obs_latency)
            # print("Deviation" , abs(obs_latency - MEDIAN_LATENCY))
            # print("---------------------------")
            if len(self.memory) > self.batch_size:
                self.experience_replay(self.batch_size)

            return reward