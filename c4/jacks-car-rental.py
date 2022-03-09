import numpy as np
import math

REQ_LOC1_LAMBDA = 3
REQ_LOC2_LAMBDA = 4
RET_LOC1_LAMBDA = 3
RET_LOC2_LAMBDA = 2
GAMMA = 0.9
MAX_ITERS = 3
MAX_CAR_NUM = 20
THETA = 0.1

policies = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1), dtype=int)
values = np.zeros((MAX_CAR_NUM + 1, MAX_CAR_NUM + 1))

def prob(n, lambd):
  return ((lambd ** n) * math.exp(-lambd)) / (math.prod(range(1, n + 1)))

def compute_value(cars_loc1, cars_loc2, a):
  v_new = 0
  # move cars
  loc1_moved = cars_loc1 - a
  loc2_moved = cars_loc2 + a
  for rented_loc1 in range(0, REQ_LOC1_LAMBDA * 2 + 1, 1):
    for rented_loc2 in range(0, REQ_LOC2_LAMBDA * 2 + 1, 1):
      # each possibility of rented car numbers at location 1 and 2 (0 - 2 * mu)
      for returned_loc1 in range(0, RET_LOC1_LAMBDA * 2 + 1, 1):
        for returned_loc2 in range(0, RET_LOC2_LAMBDA * 2 + 1, 1):
          # add up returned cars in each location ( 0 - 2 * mu)
          probability = \
            prob(rented_loc1, REQ_LOC1_LAMBDA) * \
            prob(rented_loc2, REQ_LOC2_LAMBDA) * \
            prob(returned_loc1, RET_LOC1_LAMBDA) * \
            prob(returned_loc2, RET_LOC2_LAMBDA)
          reward = min(loc1_moved, rented_loc1) * 10 \
            + min(loc2_moved, rented_loc2) * 10 \
            - abs(a) * 2
          remained_loc1 = min(loc1_moved - min(loc1_moved, rented_loc1) + returned_loc1, MAX_CAR_NUM)
          remained_loc2 = min(loc2_moved - min(loc2_moved, rented_loc2) + returned_loc2, MAX_CAR_NUM)
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
        # for each action
        a = policies[cars_loc1][cars_loc2]
        # update value of current state
        new_v = compute_value(cars_loc1, cars_loc2, a)
        values[cars_loc1][cars_loc2] = new_v
        delta = abs(v - new_v)
    print('Delta', delta)
    if (delta < THETA):
      break
  return None

eval_policy()

# policy improvement
for i in range(MAX_ITERS): 
  print('Policy improvement, iteration', i)
  # value iteration: one iteration
  policy_stable = True
  # for each state
  for cars_loc1 in range(MAX_CAR_NUM + 1):
    for cars_loc2 in range(MAX_CAR_NUM + 1):
      a = policies[cars_loc1][cars_loc2]
      new_values = []
      max_cars_to_move = min(min(cars_loc1, 5), MAX_CAR_NUM - cars_loc2)
      min_cars_to_move = - min(min(cars_loc2, 5), MAX_CAR_NUM - cars_loc1)
      # for each action
      for a in range(min_cars_to_move, max_cars_to_move + 1, 1):
        # update value of current state
        new_values.append(compute_value(cars_loc1, cars_loc2, a))
      if (len(new_values) > 0):
        pi = np.argmax(new_values) + min_cars_to_move
        policies[cars_loc1][cars_loc2] = pi
      if (a != pi):
        policy_stable = False
  if (policy_stable):
    break
  else:
    eval_policy()

print(values)
print(policies)
