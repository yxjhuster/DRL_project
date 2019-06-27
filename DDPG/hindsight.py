import random
from IPython import embed

K_HINDSIGHT = 4

def hindsight(state, goalx=0, goaly=0):
    return_state = state[:]
    return_state.append(goalx)
    return_state.append(goaly)
    return return_state

def strategy_future(transitions_this_epi, t):
    additional_goals = []
    try:
        transitions = transitions_this_epi[t+1:]
    except:
        return additional_goals
    try:
        transition_batch = random.sample(transitions, K_HINDSIGHT)
    except:
        transition_batch = random.sample(transitions, int(len(transitions)/2))
    
    for i in range(len(transition_batch)):
        additional_goals.append([transition_batch[i][0][0][6],  # deltax
                                 transition_batch[i][0][0][7]]) # deltay

    return additional_goals