import numpy as np
import math
import matplotlib.pyplot as plt

MAX_CAPITAL = 100
P_HEAD = 0.4 # probability of head
GAMMA = 1
MAX_ITERS = 300

values = np.zeros(100)
policies = np.zeros(100, dtype=int)

def compute_value(s, a):
  v_new = P_HEAD * (1 if s + a >= 100 else GAMMA * values[s + a]) + \
          (1 - P_HEAD) * (GAMMA * values[s - a] if s > a else 0)
  return v_new

# calculate value[s] for each possible actions, 
#   find best action value pair, and 
#   return delta of value change
def eval_value(s):
  v = values[s]
  action_values = []
  for a in range(0, s + 1):
    # for each possible bets
    action_values.append(compute_value(s, a))
  if len(action_values) > 0:
    values[s] = np.max(action_values)
  return abs(values[s] - v)

def argmax_last(value_array):
  max_value = max(value_array)
  final_index = value_array.index(max_value) 
  # final_index = max(index for index, item in enumerate(value_array) if item == max_value)
  return final_index

def eval_policy(s):
  action_values = []
  for a in range(0, s + 1):
    # for each possible bets
    action_values.append(compute_value(s, a))
  if len(action_values) > 0:
    policies[s] = argmax_last(action_values)
    # print(policies[s], action_values)
  return None

for k in range(MAX_ITERS):
  delta = 0
  for s in range(1, MAX_CAPITAL):
    delta = max(delta, eval_value(s))
  print('Delta = ', delta, 'for iteration', k)
  if delta < 0.000001:
    break

for s in range(1, MAX_CAPITAL):
 eval_policy(s)

print(values)

# eval_policy(20)
print(policies)

plt.plot(range(MAX_CAPITAL), policies)
plt.show()
