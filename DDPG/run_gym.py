'''
Reference https://github.com/lightaime/TensorAgent/blob/master/
'''
from model.ddpg_model import Model
from agent.ddpg import Agent
from mechanism.replay_buffer import Replay_Buffer
from mechanism.ou_process import OU_Process
from gym import wrappers
import gym, copy
import numpy as np
from IPython import embed
from KukaEnv_10703 import KukaVariedObjectEnv
from IPython import embed
from plot import *
from hindsight import *
import argparse

EPISODES = 100000
MAX_EXPLORE_EPS = 100
STEP_LIMIT_EPI = 1500
TEST_EPS = 1
BATCH_SIZE = 64
BUFFER_SIZE = 1e6
DISCOUNT_FACTOR = 0.99
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
TAU = 0.001
NUMOBJECTS = 9
METHOD_NAME = 'DDPG_PURE'  
#'DDPG_HINDSIGHT_XYCOST'  
#'DDPG_INTRINSIC_REWARD'  
#'DDPG_HINDSIGHT_XYCOST_INTRINSIC_REWARD'

# This function is adapted from the one in homework 3
def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--items-path', dest='items_path',
                        type=str, default='/home/calvinqiao/Documents/Courses/Deep Reinforcement Learning and Control/Project-Newest/10703-Manipulation-Env/TensorAgent/items', help="Path to the items file.")
    # parser.add_argument('--ddpg-method', dest='ddpg_method', type=str,
                        # default='DDPG_PURE', help="Select from DDPG PURE, DDPG_HINDSIGHT_XYCOST, DDPG_INTRINSIC_REWARD, DDPG_HINDSIGHT_XYCOST_INTRINSIC_REWARD.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # parser_group = parser.add_mutually_exclusive_group(required=False)

    return parser.parse_args()

if 'HINDSIGHT' in METHOD_NAME:
    WARM_UP_MEN = (5**2) * BATCH_SIZE
else:
    WARM_UP_MEN = 5 * BATCH_SIZE

def onehot(state, env):
    obj_idx = state[-1]
    onehot = [0] * NUMOBJECTS
    onehot[obj_idx] = 1 
    state.pop(-1)
    state.extend(onehot)
    state.append(env.get_object_z())
    return state

def main():
    args = parse_arguments() 

    # Make env
    item_path = args.items_path
    env = KukaVariedObjectEnv(item_path, renders=False, 
                              isDiscrete=False, removeHeightHack=False, 
                              maxSteps=STEP_LIMIT_EPI)

    
    
    # Init
    if 'HINDSIGHT' in METHOD_NAME and 'XYCOST' in METHOD_NAME:
        state_dim = 21
    elif METHOD_NAME == 'DDPG_INTRINSIC_REWARD':
        state_dim = 19
    else:
        state_dim = 19
    action_dim = env.action_space.shape[0]
    model = Model(state_dim,
                  action_dim,
                  actor_learning_rate=ACTOR_LEARNING_RATE,
                  critic_learning_rate=CRITIC_LEARNING_RATE,
                  tau=TAU)
    replay_buffer = Replay_Buffer(buffer_size=int(BUFFER_SIZE) ,batch_size=BATCH_SIZE)
    exploration_noise = OU_Process(action_dim)
    agent = Agent(model, replay_buffer, exploration_noise, discout_factor=DISCOUNT_FACTOR)

    action_mean = 0
    reward_list = []
    i, check = 0, 0

    # Start episodes
    for episode in range(EPISODES):
        state = env.reset()
        state = onehot(env.get_feature_vec_observation(), env)
        if 'HINDSIGHT' in METHOD_NAME:
            state = hindsight(state)
        
        agent.init_process()
        reward_epi = 0  # Keep track of the real reward for this episode
        transitions_this_epi, reward_this_epi = [], []

        # Training:
        for step in range(STEP_LIMIT_EPI):
            state = np.reshape(state, (1, -1))
            
            # Get action
            if episode < MAX_EXPLORE_EPS:
                p = episode / MAX_EXPLORE_EPS
                action = np.clip(agent.select_action(state, p), -1.0, 1.0)
            else:
                action = agent.predict_action(state)
            action_ = action[0]
            
            next_state, reward, reward_train, done, _ = env.step(action_)
            reward_epi += reward
            next_state = onehot(env.get_feature_vec_observation(), env)
            if 'HINDSIGHT' in METHOD_NAME:
                next_state = hindsight(next_state)
            next_state = np.reshape(next_state, (1, -1))

            if METHOD_NAME == 'DDPG_HINDSIGHT_XYCOST':
                agent.store_transition([state, action, reward_train, next_state, done])
                transitions_this_epi.append([state, action, reward_train, next_state, done])
            elif METHOD_NAME == 'DDPG_HINDSIGHT_XYCOST_INTRINSIC_REWARD':
                agent.store_transition([state, action, reward_train, next_state, done])
                transitions_this_epi.append([state, action, reward_train, next_state, done])
            elif METHOD_NAME == 'DDPG_INTRINSIC_REWARD':
                agent.store_transition([state, action, reward_train, next_state, done])                 
            else:
                agent.store_transition([state, action, reward, next_state, done])

            reward_this_epi.append(reward)

            if agent.replay_buffer.memory_state()["current_size"] > WARM_UP_MEN:
                agent.train_model()
                if check == 0: 
                    print('Start training')
                    check += 1
            else:
                i += 1
                action_mean = action_mean + (action - action_mean) / i
            
            state = next_state
            if done:
                break
        reward_list.append(reward_epi)
        print('Episode step', episode, ' Reward at this episode', reward_epi)
        
        if 'HINDSIGHT' in METHOD_NAME:
            T = len(transitions_this_epi)
            for t in range(T):
                state_t, action_t, reward_t, next_state_t, done_t = transitions_this_epi[t]
                # Sample a set of goals for replay G := S(current_episode)
                additional_goals = strategy_future(transitions_this_epi, t)
                # Add additional transitions using new goals
                for i in range(len(additional_goals)):
                    new_goal_x = additional_goals[i][0]
                    new_goal_y = additional_goals[i][1]
                    new_reward, new_reward_train = env.get_reward([next_state_t[0][6], next_state_t[0][7]], 
                                                [new_goal_x, new_goal_y], next_state_t[0][-3])
                    state[0][-2], next_state[0][-2] = new_goal_x, new_goal_x
                    state[0][-1], next_state[0][-1] = new_goal_y, new_goal_y
                    agent.store_transition([state_t, action_t, 
                                            new_reward, next_state_t, done_t])

    plot(reward_list, METHOD_NAME)

if __name__ == '__main__':
    main()
