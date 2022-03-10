import numpy as np
import math

AVG_REQ1 = 3
AVG_REQ2 = 4
AVG_RET1 = 3
AVG_RET2 = 2
GAMMA = 0.9
MAX_ITERS = 10
MAX_CAR_NUM = 20
THETA = 5

policies = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1), dtype=int)
values = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))

def prob(n, lambd):
  return ((lambd ** n) * math.exp(-lambd)) / (math.prod(range(1, n + 1)))

def compute_value(cars_loc1, cars_loc2, a):
  v_new = 0
  # available cars at locations 1 and 2 after moving action
  available_1 = cars_loc1 - a
  available_2 = cars_loc2 + a
  for requested_1 in range(0, AVG_REQ1 * 4 + 1, 1):
    for requested_2 in range(0, AVG_REQ2 * 4 + 1, 1):
      # each possibility of rented car numbers at location 1 and 2 (0 - 2 * mu)
      for returned_loc1 in range(0, AVG_RET1 * 4 + 1, 1):
        for returned_loc2 in range(0, AVG_RET2 * 4 + 1, 1):
          # add up returned cars in each location ( 0 - 2 * mu)
          probability = \
            prob(requested_1, AVG_REQ1) * \
            prob(requested_2, AVG_REQ2) * \
            prob(returned_loc1, AVG_RET1) * \
            prob(returned_loc2, AVG_RET2)
          reward = min(available_1, requested_1) * 10 \
            + min(available_2, requested_2) * 10 \
            - abs(a) * 2
          remained_loc1 = min(available_1 - min(available_1, requested_1) + returned_loc1, MAX_CAR_NUM)
          remained_loc2 = min(available_2 - min(available_2, requested_2) + returned_loc2, MAX_CAR_NUM)
          v_new += probability * (reward + \
            GAMMA * values[remained_loc1][remained_loc2])
  return v_new


def eval_policy():
  # policy evaluation
  for i in range(MAX_ITERS): 
    print('Policy Evaluation, iteration', i)
    # value iteration: one iteration
    delta = 0
    # for each state
    for cars_loc1 in range(MAX_CAR_NUM + 1):
      for cars_loc2 in range(MAX_CAR_NUM + 1):
        v = values[cars_loc1][cars_loc2]
        # follow current policy
        a = policies[cars_loc1][cars_loc2]
        # update value of current state
        new_v = compute_value(cars_loc1, cars_loc2, a)
        values[cars_loc1][cars_loc2] = new_v
        delta = max(delta, abs(v - new_v))
    print('Delta', delta)
    if (delta < THETA):
      break
  return None

def improve_policy():
  # policy improvement
  for i in range(MAX_ITERS): 
    print('Policy improvement, iteration', i)
    # value iteration: one iteration
    policy_stable = True
    unstable = 0
    # for each state
    for cars_loc1 in range(MAX_CAR_NUM + 1):
      for cars_loc2 in range(MAX_CAR_NUM + 1):
        pi_old = policies[cars_loc1][cars_loc2]
        new_values = []
        max_cars_to_move = min(min(cars_loc1, 5), MAX_CAR_NUM - cars_loc2)
        min_cars_to_move = - min(min(cars_loc2, 5), MAX_CAR_NUM - cars_loc1)
        # for each action
        for a in range(min_cars_to_move, max_cars_to_move + 1, 1):
          # update value of each action
          new_values.append(compute_value(cars_loc1, cars_loc2, a))
        if (len(new_values) > 0):
          # find the optimal: argmax
          pi = np.argmax(new_values) + min_cars_to_move
          policies[cars_loc1][cars_loc2] = pi
        if (pi_old != pi):
          policy_stable = False
          unstable += 1
    if (policy_stable):
      break
    else:
      print('Unstable policies:', unstable)
      eval_policy()
  return None

eval_policy()

improve_policy()

print(values)
print(policies)
