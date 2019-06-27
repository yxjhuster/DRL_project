import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras import backend as bk
import gym

import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gym, sys, copy, argparse, random, time

from KukaEnv_10703 import KukaVariedObjectEnv

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Qnetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	# Input  - state of the world
	# Output - Q values of the actions available to the agent

	def __init__(self, state_dimension, action_dimension):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here. 

		learning_rate = 0.001
		self.qnetwork_optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

		# Define network architecture, Dueling DQN (including value stream and advantage stream)
		self.input_layer = Input(shape=(state_dimension,))
		self.hidden_layer_value1 = Dense(128, activation='relu')(self.input_layer)
		self.hidden_layer_value2 = Dense(128, activation='relu')(self.hidden_layer_value1)
		self.value_layer = Dense(1, activation='linear')(self.hidden_layer_value2)
		self.hidden_layer_advantage1 = Dense(128, activation='relu')(self.input_layer)
		self.hidden_layer_advantage2 = Dense(128, activation='relu')(self.hidden_layer_advantage1)
		self.advantage_layer = Dense(action_dimension, activation='linear')(self.hidden_layer_advantage2)
		
		self.value_layer = Lambda(lambda value: tf.tile(value, [1, action_dimension]))(self.value_layer)
		self.advantage_layer = Lambda(lambda advantage: advantage-tf.reduce_mean(advantage, axis=-1, keepdims=True))(self.advantage_layer)
		self.output_layer = keras.layers.add([self.value_layer, self.advantage_layer])
		
		self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
		self.model.compile(loss='mse', optimizer=self.qnetwork_optimizer)

		self.model_file, self.weight_file = 'model_DuelingDQN.json', 'model_DuelingDQN.h5'

	def save_model(self):
		model_file = self.model_file
		with open(model_file,'w') as json_file:
			json_string = self.model.to_json()
			json_file.write(json_string)
		json_file.close()
		self.save_model_weights()

	def save_model_weights(self):
		# Helper function to save your model and model weights. 
		weight_file = self.weight_file
		self.model.save_weights(weight_file)

	def load_model(self):
		# Helper function to load an existing model and its model weights.
		model_file = self.model_file
		with open(model_file,'r') as json_file:
			model_json = json_file.read()
			self.model = model_from_json(model_json, custom_objects={'tf': tf})
		json_file.close()
		self.load_model_weights()
		
	def load_model_weights(self):
		# Helper funciton to load model weights. 
		weight_file = self.weight_file
		self.model.load_weights(weight_file)


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000): #!
		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 

		self.transition_memory = []
		self.memory_size = memory_size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=64):
		# This function returns a batch of randomly sampled transitions 
		# - i.e. state, action, reward, next state, terminal flag tuples
		# You will feed this to your model to train.
		return random.sample(self.transition_memory, batch_size)

	def append(self, transition):

		if len(self.transition_memory) > self.memory_size:
			self.transition_memory.pop(0)
			# Appends transition to the memory
			self.transition_memory.append(transition)
		else:
			# Appends transition to the memory
			self.transition_memory.append(transition)


class DQN_agent():
	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, env):
		self.env = env
		self.env.reset()

		# setup environment parameters
		self.state_dimension = len(self.env.get_feature_vec_observation())
		self.action_dimension = self.env.action_space.n

		# Training parameters
		self.num_episodes = 50000 #!
		self.num_ploting_episode = 1000 #!
		self.num_test_episodes = 50 #!

		self.mean_list = []
		self.std_list = []
		self.reward_list = []
		self.iter_count = 0

        # Net work parameters
		self.target_qnetwork_update_rate = 100
		self.qnetwork = Qnetwork(self.state_dimension, self.action_dimension)
		self.qnetwork_target = Qnetwork(self.state_dimension, self.action_dimension)
		# define memory replayer
		self.memory_replayer = Replay_Memory()

		# define gamma
		self.gamma = 0.99

		# define epsilon
		self.epsilon = 0.5  
		self.epsilon_min = 0.05 
		self.epsilon_update_rate = (self.epsilon-self.epsilon_min) / 100000


	def epsilon_greedy_policy(self, qvalues):
		# Creating epsilon greedy probabilities to sample from. 
		# With probability 1-epsilon, A = argmax_a[Q(a)]
		# With probability epsilon, A = a random action
		if np.random.random() < self.epsilon:
			action = self.env.action_space.sample()
			return action
		else:
			# Get optimal action
			return np.argmax(qvalues)

	def greedy_policy(self, qvalues):
		# Creating greedy policy for test time. 
		# Get optimal action
		return np.argmax(qvalues)

	def adjust_obs_format(self, observation):
		observation = np.array(observation)
		return observation.reshape(1, self.state_dimension)


	def train(self):

		print("Start Training!")

		self.burn_in_memory()

		for idx_episode in range(self.num_episodes):
			self.env.reset()
			observation = self.env.get_feature_vec_observation()
			total_reward = 0
			while True:
				# print(observation)
				qvalues = self.qnetwork.model.predict(self.adjust_obs_format(observation))

				# Get an action a_t using epsilon_greedy_policy
				action = self.epsilon_greedy_policy(qvalues) 

				# Execute an action and observe reward r_t and new state
				_, reward, done, info = self.env.step(action)
				new_observation = self.env.get_feature_vec_observation()

				# Store transition to the replay_memory
				self.memory_replayer.append((observation, action, reward, done, new_observation))	
				total_reward += reward

				# Sample random minibatch of transitions from the replay_memory of transitions
				minibatch = self.memory_replayer.sample_batch()

				# Extract corresponding inputs(states), yvalues(target Q values)
				inputs = np.array([elem[0] for elem in minibatch])
				yvalues = np.array(self.get_yvalues(minibatch))

				history = self.qnetwork.model.fit(inputs, yvalues, verbose=0)
				# print('training success')
				if self.iter_count % self.target_qnetwork_update_rate == 0:
					new_weights = self.qnetwork.model.get_weights()
					self.qnetwork_target.model.set_weights(new_weights)

				# if idx_episode % self.num_ploting_episode == 0:
				# 	self.test_training()


				self.epsilon = np.max([0.05, self.epsilon-self.epsilon_update_rate]) 

				self.iter_count += 1

				# Update observation
				observation = new_observation

				# Check if the next state is terminal
				if done: break
			if idx_episode % self.num_ploting_episode == 0:
				self.test_training()
			self.reward_list.append(total_reward)
			print("Episode: %d, total reward: %f, loss: %f" % (idx_episode, total_reward, history.history['loss'][-1]))
		# print(self.reward_list)
		output = open('reward.pkl', 'wb')
		pickle.dump(self.reward_list, output)
		plot_mean("DQN", self.reward_list)
		self.plot()



	def get_yvalues(self, minibatch):
		yvalues = []
		batch_states = np.array([elem[0] for elem in minibatch])
		batch_next_states = np.array([elem[-1] for elem in minibatch])

		qvalues = self.qnetwork.model.predict(batch_states)
		qvalues_next_target_network = self.qnetwork_target.model.predict(batch_next_states)

		for idx_batch, elem in enumerate(minibatch):
			qtarget = qvalues[idx_batch]

			obs, action, reward, done, next_obs = elem

			# Update Q values
			if done:
				qtarget[action] = reward
			else:
				max_action = np.argmax(qvalues_next_target_network[idx_batch])
				qtarget[action] = reward + self.gamma*qvalues_next_target_network[idx_batch][max_action]

			yvalues.append(qtarget)

		return yvalues


	def test_training(self):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		print("test start!")
		total_reward = []

		for idx_test in range(self.num_test_episodes):
			reward_episode = 0
			self.env.reset()
			observation = self.env.get_feature_vec_observation()
			# print('Episode = ', idx_test)
			while True:
				# self.env.render()

				# if self.DQN_type == 'DQN' or self.DQN_type == 'DoubleDQN':
				qvalues = self.qnetwork.model.predict(self.adjust_obs_format(observation))
				# elif self.DQN_type == 'DuelingDQN':
				# qvalues = self.qnetwork_dueling.model.predict(self.adjust_obs_format(observation))
				# Get an action a_t using epsilon_greedy_policy
				action = self.greedy_policy(qvalues)  # greedy or epsilon greedy???

				# Execute an action and observe reward r_t and new state
				_, reward, done, info = self.env.step(action)

				new_observation = self.env.get_feature_vec_observation()

				reward_episode += reward

				observation = new_observation
				if done: break
			total_reward.append(reward_episode)

		reward_mean = np.mean(np.array(total_reward))
		self.mean_list.append(reward_mean)
		reward_std = np.std(np.array(total_reward))
		self.std_list.append(reward_std)


	def test(self):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		total_reward = []

		for idx_test in range(self.num_test_episodes):
			reward_episode = 0
			self.env.reset()
			observation = self.env.get_feature_vec_observation()
			# print('Episode = ', idx_test)
			while True:
				# self.env.render()

				# if self.DQN_type == 'DQN' or self.DQN_type == 'DoubleDQN':
				qvalues = self.qnetwork.model.predict(self.adjust_obs_format(observation))
				# elif self.DQN_type == 'DuelingDQN':
				# qvalues = self.qnetwork_dueling.model.predict(self.adjust_obs_format(observation))
				# Get an action a_t using epsilon_greedy_policy
				action = self.greedy_policy(qvalues)  # greedy or epsilon greedy???

				# Execute an action and observe reward r_t and new state
				_, reward, done, info = self.env.step(action)

				new_observation = self.env.get_feature_vec_observation()

				reward_episode += reward

				observation = new_observation
				if done: break
			total_reward.append(reward_episode)

		reward_mean = np.mean(np.array(total_reward))
		reward_std = np.std(np.array(total_reward))
		print("After %d episodes, mean is %f, std is %f" % (self.num_episodes, reward_mean, reward_std))


	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		num_collected_transition, done = 0, False
		while num_collected_transition < self.memory_replayer.burn_in:
			state = self.env.reset()
			observation = self.env.get_feature_vec_observation()
			while not done:
				# Take a random action and get reward r_t and new_observation
				action = self.env.action_space.sample()
				new_state, reward, done, info = self.env.step(action)
				new_observation = self.env.get_feature_vec_observation()
				# print(new_observation)
				# Append a transition to memory
				self.memory_replayer.append((observation, action, reward, done, new_observation))	
				num_collected_transition += 1
				observation = new_observation
				state = new_state
				
				if done: break
				if num_collected_transition >= self.memory_replayer.burn_in: break
			print(num_collected_transition)
			done = False
		print("Burn-in Finished!")


	def plot(self):
		# plot the test results including means and std.
		fig, (axes) = plt.subplots(nrows=1)
		x = np.array(range(0, self.num_episodes, self.num_ploting_episode))
		y = np.array(self.mean_list)
		stds = np.array(self.std_list)
		axes.errorbar(x, y, yerr=stds, fmt='-o')
		plt.savefig('Training_Performance.png')

def main(args):

	environment = KukaVariedObjectEnv(args[0], renders=False,isDiscrete=True, maxSteps = 1000)

	gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
	config = tf.ConfigProto(gpu_options=gpu_ops, device_count={'gpu':0}, log_device_placement=True)
	print("config is fine!")
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	dqn_agent = DQN_agent(environment)

	train, test = 1, 1

	if train:
		dqn_agent.train()
		dqn_agent.qnetwork.save_model()

	if test:
		dqn_agent.qnetwork.load_model()
		# Test it
		dqn_agent.test()


NUM_POINTS = 300.0

def plot_mean(prefix, rewards):
    x_gap = len(rewards) / NUM_POINTS
    x_vals = np.arange(0, len(rewards), x_gap).astype(int)
    rewards = np.array(rewards)

    for name, axis_label, func in \
        [('sum', 'Reward Sum (to date)', points_sum), \
         ('avg', 'Reward Average (next 100)', points_avg)]:
        y_vals = func(rewards, x_vals)
        for logscale in [True, False]:
            if logscale:
                plt.yscale('log')
            plt.plot(x_vals+1, y_vals)
            plt.xlabel('Unit of training (Actions in W1, Episodes in W2)')
            plt.ylabel(axis_label)
            plt.grid(which='Both')
            plt.tight_layout()
            plt.savefig(prefix + '_' + name + '_' + ('log' if logscale else 'lin') + '.png')
            plt.close()

def points_sum(rewards, x_vals):
    return np.array([np.sum(rewards[0:val]) for val in x_vals])

def points_avg(rewards, x_vals):
    return np.array([np.sum(rewards[val:min(len(rewards), val+100)]) \
                     /float(min(len(rewards)-val, 100)) for val in x_vals])


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Must call with file path to "items" directory')
		exit()
	print(sys.argv)
	main(sys.argv[1:])





