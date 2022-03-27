# resolve windy gridworld with SARSA
import numpy as np
import math

VERBOSE = False

WORLD_WIDTH = 10
WORLD_HEIGHT = 7
WIND_START = 3
WIND_END = 8
STRONG_WIND_START = 6
STRONG_WIND_END = 7

MAX_STEPS = 16000
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1

START_POINT = (3, 0)
END_POINT = (3, 7)

policies = {}
action_values = {}

# standard move
# actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# King's move
actions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

ACTION_NUMBER = len(actions)

def wind(state):
  s_v, s_h = state
  w = 0
  if s_h >= WIND_START and s_h <= WIND_END:
    w = 1
    if s_h >= STRONG_WIND_START and s_h <= STRONG_WIND_END:
      w = 2
  return w

def compute_dest(state, action_index):
  a = actions[action_index]
  next_v = max(0, min(WORLD_HEIGHT - 1, state[0] + a[0] + wind(state)))
  next_h = max(0, min(WORLD_WIDTH - 1, state[1] + a[1]))
  return (next_v, next_h)

def valid_actions(state):
  valid_actions = []
  for ai in range(ACTION_NUMBER):
    s = compute_dest(state, ai)
    if s != state:
      valid_actions.append(ai)
  return valid_actions

def select_action(state, epsilon):
  va = valid_actions(state)
  a = va[math.floor(np.random.random() * len(va))]
  # implement epsilon soft policy
  if state in policies:
      if np.random.random() < epsilon:
          a = va[math.floor(np.random.random() * len(va))]
      else:
          a = policies[state]
  return a

def get_action_value(state, action):
  value = 0
  if state in action_values:
    if action in action_values[state]:
      value = action_values[state][action]
  return value

def move(state, action_index):
  next_v, next_h = compute_dest(state, action_index)
  reward = 0 if (next_v, next_h) == END_POINT else -1
  return (next_v, next_h), reward

def update(state, action, q):
  # update action values
  if state not in action_values:
    action_values[state] = {}
  action_values[state][action] = q
  a_opt = action
  for a in action_values[state]:
    if action_values[state][a] > q:
      a_opt = a
  # update policy
  policies[state] = a_opt
  return None

def temporal_diff():
  # REPEAT at least N steps
  n = 0
  while n <= MAX_STEPS:
    # start an episode with s=S0
    s = START_POINT
    # select a=A0 with policy
    a = select_action(s, EPSILON)
    # for each step
    while True:
      n += 1
      # take an action, then observe reward and next state
      s_next, r = move(s, a)
      # select next action with policy (epsilon greedy)
      a_next = select_action(s_next, EPSILON)
      # compute and update action value for S, A
      q = get_action_value(s, a)
      q += ALPHA * (r + GAMMA * get_action_value(s_next, a_next) - q)
      update(s, a, q)
      # go to next state with next action
      s, a = s_next, a_next
      # print('state', s, 'action', a)
      # break if next state is terminal state
      if s == END_POINT:
        print('end of one episode after', n, 'steps')
        break
      # if n >= MAX_STEPS:
      #   break
  return None

# generate a test episode with learned policy
def run_episode(epsilon=0, verbose=False, max_steps=100):
  t = 0
  success = False
  s = START_POINT
  a = select_action(s, epsilon)
  print('Test: start from state', s, 'action', a)
  # for each step
  while True:
    s, r = move(s, a)
    a = select_action(s, epsilon)
    t += 1
    if verbose: 
      print('state', s, 'action', a)
    # break if next state is terminal state
    if s == END_POINT:
      success = True
      break
    if t > max_steps:
      print('FAILED to reach goal after', t, 'steps')
      break
  if success:
    print('GOAL reached after', t, 'steps')
  return None

temporal_diff()

if True:
  print(policies)
  print(action_values)

run_episode(epsilon=0.1, verbose=True)