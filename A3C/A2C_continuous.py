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

# action range [-1,1]

NUMOBJECTS = 9
STATE_DIM = 18
ACTION_DIM = 3

class A2C():
	def __init__(self, sess, env, state_dim, action_dim, n = 10):
		self.env = env
		self.sess = sess
		self.state_dimention = state_dim
		self.action_dimention = action_dim
		self.n = n

		# set-up training hyparameters
		self.num_episodes = 50000 #!
		self.critic_lr = 3*0.0001
		self.actor_lr = 3*0.0001
		self.gamma = 0.99
		self.scale_factor = 1

		# build network optimizer
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
		self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)

		self.std_gain = 1

		# build critic network
		self.critic_input = tf.placeholder(tf.float32, shape = (None, self.state_dimention))
		self.critic_output_labels = tf.placeholder(tf.float32)
		self.critic_hidden_layer1 = tf.layers.dense(self.critic_input, 256, tf.nn.relu)
		self.critic_hidden_layer2 = tf.layers.dense(self.critic_hidden_layer1, 256, tf.nn.relu)
		self.critic_output_layer = tf.layers.dense(self.critic_hidden_layer2, 1)

		#build actor network
		self.actor_input = tf.placeholder(tf.float32, shape = (None, self.state_dimention))
		self.actor_output = tf.placeholder(tf.float32, shape = (None, self.action_dimention))
		self.actor_advantage_values = tf.placeholder(tf.float32)
		self.actor_actions = tf.placeholder(tf.float32, shape = (None, self.action_dimention))
		self.actor_step_num = tf.placeholder(tf.float32)
		self.std = tf.placeholder(tf.float32, shape = (None, self.action_dimention))
		self.actor_hidden_layer1 = tf.layers.dense(self.actor_input, 256, tf.nn.relu)
		self.actor_hidden_layer2 = tf.layers.dense(self.actor_hidden_layer1, 256, tf.nn.relu)
		self.actor_output_layer = tf.layers.dense(self.actor_hidden_layer2, self.action_dimention)
		self.actor_predict_action = tf.clip_by_value(self.actor_output_layer, -1, 1)

		# std for action selection #! change this num could change the possibility distribution
		self.std = 0.2 * tf.ones([self.actor_step_num, self.action_dimention], tf.float32)


		# loss function for critic network
		self.critic_loss = tf.losses.mean_squared_error(self.critic_output_labels, self.critic_output_layer)

		# loss function for actor network
		self.action_mean = self.actor_output_layer
		self.dist = tf.distributions.Normal(loc = self.action_mean, scale = self.std)
		# self.action_sample = self.dist.sample()
		self.actor_prob_log = self.dist.log_prob(self.actor_actions) #! may be posible to be changed to log_prob
		self.actor_loss = -tf.reduce_mean(self.actor_prob_log * self.actor_advantage_values) #! -

		self.actor_train_op = self.actor_optimizer.minimize(self.actor_loss)
		self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss)

		# build initializer
		initializer = tf.global_variables_initializer()
		self.sess.run(initializer)

		# plot parameters
		self.num_test_episode = 100
		self.num_ploting_episode = 1000
		self.mean_list = []
		self.std_list = []

		self.reward_list = []


	def train(self):
        # Train the model using A2C method over the continuous action space
		for idx_episode in range(self.num_episodes):
			states, actions, rewards = self.generate_episode()
			# print(actions)

			total_step_num = len(states)
			# print(total_step_num)
			# print(len(actions))

			R = []
			# get the R value
			for time in range(total_step_num):
				Rt = 0
				if (time + self.n) >= total_step_num:
					V_end = 0
				else:
					V_end = self.sess.run(self.critic_output_layer, feed_dict = {
						self.critic_input: self.adjust_obs_format(states[time + self.n])
 						})
				Rt += V_end
				for idx in range(self.n):
					if (time + idx) >= total_step_num:
						Rt += 0 * (self.gamma ** idx)
					else:
						Rt += rewards[time + idx] * (self.gamma ** idx)
				R.append(Rt)

			R = np.reshape(np.array(R), [total_step_num,1])
			value_predict = self.sess.run(self.critic_output_layer, feed_dict = {
            			self.critic_input: np.vstack(states)
            			})

			advantage_value = R - value_predict / self.scale_factor
			advantage_reduced = advantage_value
			# print(advantage_reduced)
			# advantage_mean, advantage_std = np.mean(advantage_value), np.std(advantage_value)
			# advantage_reduced = (advantage_value-advantage_mean) / advantage_std
			advantage_reduced = np.array(advantage_reduced).reshape([total_step_num, 1])

			actor_loss, _, log_value = self.sess.run([self.actor_loss, self.actor_train_op, self.actor_prob_log], feed_dict ={
				self.actor_input: np.vstack(states),
				self.actor_actions: np.array(actions),
                # self.actor_R_value: R * self.scale_factor,
                # self.actor_V_value: value_predict * self.scale_factor,
				self.actor_advantage_values: advantage_reduced,
				self.actor_step_num: float(total_step_num) ,
				self.std: self.std_gain * np.ones([total_step_num,self.action_dimention])             
				})

			critic_loss, _ = self.sess.run([self.critic_loss, self.critic_train_op], feed_dict = {
				self.critic_input: np.vstack(states),
				self.critic_output_labels: R * self.scale_factor
				})

			reward_episode = np.sum(rewards)

			if self.std_gain > 0.001:
				self.std_gain -= 0.0005

			print('Episode %d, Reward %.1f, Actor Loss: %f, Critic Loss: %f, log_value: %f %f %f, advantage_reduced: %f' % 
									(idx_episode, reward_episode, actor_loss, critic_loss, log_value[0][0], log_value[0][1], log_value[0][2], advantage_reduced[0]))

			self.reward_list.append(reward_episode)

		output = open('reward.pkl', 'wb')
		pickle.dump(self.reward_list, output)

		plot_mean("DQN", self.reward_list)

		# 	if idx_episode % self.num_ploting_episode == 0: #! could be removed
		# 		self.test()

		# self.plot() #! could be removed
  #       return


  	def select_action(self, action):
  		# print(action)
  		# print(self.std_gain * np.ones([1, self.action_dimention]))
  		return np.clip(np.random.normal(action, self.std_gain * np.ones([1, self.action_dimention])[0],3),-1,1)




	def adjust_obs_format(self, observation):
		# print(observation)
		return observation.reshape(1, self.state_dimention)


	def adjust_state(self, state_observation):
		obj_idx = state_observation[-1]
		onehot = [0] * NUMOBJECTS
		onehot[obj_idx] = 1
		state_observation.pop(-1)
		state_observation.extend(onehot)
		return np.array(state_observation)


	def generate_episode(self):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
		states = []
		actions = []
		rewards = []

		self.env.reset()
		observation = self.adjust_state(self.env.get_feature_vec_observation())
		while True:
			states.append(observation)
			# action = self.sess.run(self.actor_output_layer, feed_dict={
			# 					self.actor_input: self.adjust_obs_format(observation)})
			action = self.sess.run(self.actor_output_layer, feed_dict={
								self.actor_input: self.adjust_obs_format(observation)})
			action = action[0]
			action = self.select_action(action)
			actions.append(action)
			_, reward, done, info = self.env.step(action)
			rewards.append(reward)

			next_observation = self.adjust_state(self.env.get_feature_vec_observation())
			observation = next_observation

			if done: break
		return states, actions, rewards

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





def main(args):
	# Basic set-up for tensorflow
	gpu_ops = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
	config = tf.ConfigProto(gpu_options=gpu_ops, device_count={'gpu':0}, log_device_placement=True)
	sess = tf.Session(config=config)
	# sess = tf.Session()
	keras.backend.tensorflow_backend.set_session(sess)
	# Environment setup
	items_path = '/home/xinjiay/Documents/10_703_project/A2C/items'
	env = KukaVariedObjectEnv(items_path, renders=False, 
                              isDiscrete=False, removeHeightHack=False, 
                              maxSteps=1000)
	a2c_agent = A2C(sess, env, STATE_DIM, ACTION_DIM, n = 10)
	a2c_agent.train()


if __name__ == '__main__':
	main(sys.argv)