import numpy as np
import math

TRACK = np.flip(np.array([
            [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5],  # additional safety 
            [-5, -5, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],  # track 1 end with finish line
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0], 
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0], 
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],  
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5],  # rect 2
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  # rect 3, 7 lines 
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  # rect 4, 8 lines
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -1, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  # rect 5, 7 lines
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -1, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -5, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5],  # rect 6 from start line
            [-5, -5, -5, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
            [-5, -5, -5, -1, -1, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5, -5, -5], 
        ]), 0)
TRACK_DIM = TRACK.shape
ACTIONS = np.array([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]])
MAX_VELOCITY = 5
TRAINING_EPISODES = 50000
EPS = 0.5

# policies = np.ones(TRACK_DIM, dtype=int) * 8 # same dimension as track
policies = {}
# values = np.zeros() # state values
action_values = {} # action value q(s, a)

def average(returns):
    average = 0
    if (len(returns) > 0):
        average = float(sum(returns) / len(returns))
    return average

def adjust_speed(v, a):
    v_new = v + ACTIONS[a]
    v_new[0] = min(v_new[0], 5)
    v_new[1] = min(v_new[1], 5)
    v_new[0] = max(v_new[0], 0)
    v_new[1] = max(v_new[1], 0)
    if (np.array_equal(v_new, [0, 0])):
        # possible velocity turn still: [0, 1] [1, 0] [1, 1]
        # ** [0, 0] is not a possible
        if (np.array_equal(v, [1, 1])):
            v_new = np.array([0, 1]) if np.random.rand() >= 0.5 else np.array([1, 0])
        else:
            v_new = v
    return v_new

def race(s, v):
    # print('Race from position', s, 'with speed', v)
    r = -1
    finished = False
    v_togo, h_togo = v[0], v[1]
    v_cur, h_cur = s[0], s[1]
    # move vertical
    while (v_togo > 0 or h_togo > 0):
        if finished:
            break
        for _ in range(v_togo):
            target = TRACK[v_cur + 1][h_cur]
            if (target == -5):
                # if out of track
                r += -5
                h_togo += 1
            elif (target == 0):
                # reach finish line
                v_cur += 1
                finished = True
                break
            else:
                # move one grid vertically
                v_cur += 1
        v_togo = 0
        if  finished:
            break
        # move horizonal then
        for _ in range(h_togo):
            target = TRACK[v_cur][h_cur + 1]
            if (target == -5):
                # if out of track, go vertical instead
                r += -5
                v_togo += 1
            elif (target == 0):
                # reach finish line
                h_cur += 1
                finished = True
                break
            else:
                # move one grid vertically
                h_cur += 1
        h_togo = 0
    return v_cur, h_cur, r, finished

def epsilon_soft(state, epsilon):
    # implement epsilon soft policy
    if state in policies:
        if np.random.random() >= (1 - epsilon):
            a = math.floor(np.random.random() * 9)
            # print('exploring policy for', state, 'with', a, 'instead of greedy policy', policies[state])
        else:
            a = policies[state]
    else:
        a = 0 # math.floor(np.random.random() * 9)
        # policies[state] = a
    return a

# generate an episode
def gen_episode(h_start, greedy=False):
    # random select a start point
    s = [0, h_start]
    # initial velocity
    v = [1, 0]
    # initial return
    g = 0
    # initial trajectory
    episode = {}
    while True:
        # calculate state index
        state_index = (s[0], s[1], v[0], v[1])
        # pick action if in policy, otherwise apply a random policy
        a = epsilon_soft(state_index, 0 if greedy else EPS)
        # adjust speed
        v_next = adjust_speed(v, a)
        # calculate next position
        y_next, x_next, reward, finished = race(s, v_next)
        g += reward
        episode[state_index] = {}
        episode[state_index][a] = g
        # break if get finish line
        if finished:
            break
        v = v_next
        s = [y_next, x_next]
    return episode

# policy evaluation w/ Monte Carlo
def monte_carlo():
    # run episodes as defined
    for k in range(TRAINING_EPISODES):
        if k % 1000 == 0:
            print('going through', k, 'th episode')
        # generate episode from random start point
        h_start = math.floor(np.random.random() * 6 + 3)
        episode = gen_episode(h_start)
        # find last return
        # print(episode)
        q_last = episode[list(episode)[-1]]
        final_ret = q_last[list(q_last)[-1]]
        # for each (s, a) = g
        for state in episode:
            for action in episode[state]:
                g = final_ret - episode[state][action]
                if state in action_values:
                    if action in action_values[state]:
                        action_values[state][action].append(g)
                    else:
                        action_values[state][action] = [g]
                else:
                    action_values[state] = {}
                    action_values[state][action] = [g]

        # policy improvement: for each state s, find best action a
        for state in action_values:
            actions = []
            values = []
            for action in action_values[state]:
                actions.append(action)
                values.append(average(action_values[state][action]))
            if len(actions) > 0:
                policies[state] = actions[np.argmax(values)]
    return None

def temporal_diff():
    #
    return None

monte_carlo()
# print('action values: ', action_values)
# print('policies', policies)

# TEST with greedy policy
print('******************** Generate a TEST episode with GREEDY policy ********************')
for start_point in range(4, 9):
    episode = gen_episode(start_point, True)
    print(episode)
