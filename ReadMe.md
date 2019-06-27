This package consists of three parts (python3 environment):
1. DQN based method
In DQN folder.
- Based on and adapted from homework 2.
- DQN method should be executed with the original Kuka pybullet environment.
- Set --items-path='absolute_path_to_items_folder' to pass in the path of the items folder.
- In DQN_data folder, the training data for 50000 episodes using DQN method is saved as pkl file.

2. A2C based (continuous action) method
In A2C folder.
- Based on and adapted from homework 3.
- A2C method should be executed with the original Kuka pybullet environment.
- Set --items-path='absolute_path_to_items_folder' to pass in the path of the items folder.
- In A2C_data folder, the training data for 50000 episodes using A2C is saved as pkl file for creating plots.

3. DDPG based method
In DDPG folder, this folder includes the following methods.
- Set METHOD_NAME in run_gym.py to set the specific DDPG method
- Methods include 'DDPG_PURE', 'DDPG_HINDSIGHT_XYCOST', 'DDPG_INTRINSIC_REWARD', 'DDPG_INTRINSIC_REWARD_HINDSIGHT_XYCOST'.
- Set --items-path='absolute_path_to_items_folder' to pass in the path of the items folder.
- Run run_gym.py to start training.
- Important things to notice for using DDPG based method:
  kuka_diverse_object_gym_env.py is modified. In this script, get_reward method (line 297-325) and _reward method (line 327-379) need to set properly to correspond to the specific DDPG method chosen in the run_gym/py script, more specifically, by commenting or uncommenting certain lines in these two methods.
  1) These two methods both have four types of rewards to choose from: hindsight reward, intrinsic reward simple, intrinsic reward complex, regular task reward.
  2) get_reward is used for computing new rewards in the replay process in the Hindsight Experience Replay method.
  3) _reward is more general and used in the step function.
  4) Need to make sure that these two methods are consistent in how the reward is computed.
- In DDPG/saved_rewards folder, final results of using DDPG, DDPG+HER, DDPG+Intrinsic_Simple, DDPG+Intrinsic_Complex, and DDPG+HER+Intrinsic_Simple are saved as pkl files for plotting. The final plot that compares different methods can be created by running plot.py.
- DDPG module is based on https://github.com/lightaime/TensorAgent/blob/master. (Special thanks to the original author)


