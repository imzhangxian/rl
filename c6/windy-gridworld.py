# resolve windy gridworld with SARSA
import numpy as np
import math

WORLD_WIDTH = 10
WORLD_HEIGHT = 7

MAX_EPISODES = 100
EPSILON = 0.1
ALPHA = 0.3
GAMMA = 0.7

START_POINT = (0, 3)
END_POINT = (7, 3)

policies = []
action_values = []

actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def wind(state):
  s_v, s_h = state
  w = 0
  if s_h >= 3 and s_h <= 8:
    w = 1
    if s_h == 6 or s_h == 7:
      w = 2
  return w

def move(state, action):
  next_s = state + action + (wind(state), 0)
  next_v, next_h = next_s
  next_v = max(0, min(WORLD_HEIGHT - 1, next_v))
  next_h = max(0, min(WORLD_WIDTH, WORLD_WIDTH))
  reward = 0 if (next_v, next_h) == END_POINT else -1
  return (next_v, next_h), reward

def select_action(state, epsilon):
  # implement epsilon soft policy 
  # (TODO check if same as epsilon greedy)
  if state in policies:
      if np.random.random() >= (1 - epsilon):
          a = math.floor(np.random.random() * 4)
      else:
          a = policies[state]
  else:
      a = 0 # math.floor(np.random.random() * 4)
  return a

def temporal_diff():
  # REPEAT
  for n in range(MAX_EPISODES):
    # start an episode with s=S0
    s = (0, 3) # always start from middle
    # select a=A0 with policy
    a = select_action(s, EPSILON)
    # for each step
    while True:
      # take an action, then observe reward and next state
      s_next, r = move(s, a)
      # select next action with policy (epsilon greedy)
      a_next = select_action(s, EPSILON)
      # compute and update action value for S, A
      action_values[(s, a)] = action_values[(s, a)] + \
        ALPHA * (r + GAMMA * action_values[(s_next, a_next)] - action_values[(s, a)])
      # go to next state with next action
      s, a = s_next, a_next
      # break if next state is terminal state
      if s == END_POINT:
        break
  return None

# generate a test episode with learned policy