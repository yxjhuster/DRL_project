import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from IPython import embed

NUM_POINTS = 300.0

# Plot one method
def plot(rewards, method_name):
    pickle.dump(rewards, open(method_name+'.p', 'wb'))
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
            plt.savefig(name + '_' + ('log' if logscale else 'lin') + '_' + method_name + '_.png')
            plt.close()

def points_sum(rewards, x_vals):
    return np.array([np.sum(rewards[0:val]) for val in x_vals])

def points_avg(rewards, x_vals):
    return np.array([np.sum(rewards[val:min(len(rewards)-1, val+100)])/100 \
                     for val in x_vals])

# Plot all methods
def plot_(rewards_collection):
    for rewards in rewards_collection:
        x_gap = len(rewards) / NUM_POINTS
        x_vals = np.arange(0, len(rewards), x_gap).astype(int)
        rewards = np.array(rewards)

        for name, axis_label, func in \
            [\
             ('avg', 'Reward Average (next 100)', points_avg)]:
            y_vals = func(rewards, x_vals)
            logscale = False
            for logscale in [False]:
                if logscale:
                    plt.yscale('log')
                plt.plot(x_vals+1, y_vals)
                plt.xlabel('Unit of training (Actions in W1, Episodes in W2)')
                plt.ylabel(axis_label)
                plt.grid(which='Both')
                plt.tight_layout()

    plt.legend(('DDPG', 'DDPG Intrinsic Reward - 1', 'DDPG Intrinsic Reward - 2', 
                'DDPG HER', 'DDPG HER + Intrinsic Reward'), \
                loc='lower right')
    plt.savefig(name + '_' + ('log' if logscale else 'lin') + '_' + 'all' + '.png')
    plt.close()

    for rewards in rewards_collection:
        x_gap = len(rewards) / NUM_POINTS
        x_vals = np.arange(0, len(rewards), x_gap).astype(int)
        rewards = np.array(rewards)

        for name, axis_label, func in \
            [\
             ('sum', 'Reward Sum (to date)', points_sum)]:
            y_vals = func(rewards, x_vals)
            logscale = False
            for logscale in [False]:
                if logscale:
                    plt.yscale('log')
                plt.plot(x_vals+1, y_vals)
                plt.xlabel('Unit of training (Actions in W1, Episodes in W2)')
                plt.ylabel(axis_label)
                plt.grid(which='Both')
                plt.tight_layout()

    plt.legend(('DDPG', 'DDPG Intrinsic Reward - 1', 'DDPG Intrinsic Reward - 2',
                'DDPG HER', 'DDPG HER + Intrinsic Reward'), \
                loc='lower right')
    plt.savefig(name + '_' + ('log' if logscale else 'lin') + '_' + 'all' + '.png')
    plt.close()



def plot_all():
    reward_ddpg_pure = pickle.load(open('./saved_rewards/DDPG.pkl', 'rb'))
    reward_ddpg_intrinsic = pickle.load(open('./saved_rewards/DDPG_INTRINSIC_SIMPLE.p', 'rb'))
    reward_ddpg_intrinsic_xyz = pickle.load(open('./saved_rewards/DDPG_INTRINSIC_COMPLEX.pkl', 'rb'))
    reward_ddpg_hindsight = pickle.load(open('./saved_rewards/DDPG_HER.pkl', 'rb'))
    reward_both = pickle.load(open('./saved_rewards/DDPG_BOTH.pkl', 'rb'))

    rewards_collection = [reward_ddpg_pure,
                          reward_ddpg_intrinsic,
                          reward_ddpg_intrinsic_xyz,
                          reward_ddpg_hindsight,
                          reward_both]
    plot_(rewards_collection)

def main():
    plot_all()


if __name__ == '__main__':
    main()