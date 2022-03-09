import numpy as np
import math

# n-bandit is a stateless problem that rewards only depend on action
# no state or value function required
NUM_ARMS = 10
TOTAL_STEPS = 2000
EPS = 0.1

# placeholders for q/Q/Nt
q = np.zeros(NUM_ARMS)
Q = np.zeros(NUM_ARMS)
Nt = np.zeros(NUM_ARMS)

# stationary rewards: initialize q(a) by normal distribution
# for i in range(NUM_ARMS):
#  q[i] = np.random.normal()

# select an arm with epsilon-greedy on step t
def epsilonGreedy(t): 
  action = np.argmax(Q)
  thres = np.random.random()
  if thres < EPS:
    # random select non-optimal action
    sample = math.floor(np.random.random() * 9)
    if sample >= action:
      sample += 1
    action = sample
  return action

# take an arm and return reward, per current Q
def bandit(a): 
  # for question 2.4: all rewards in random walk
  for i in range(NUM_ARMS):
    q[i] += (1 if np.random.random() >= 0.5 else -1)
  # create noise
  noise = np.random.normal()
  # noise = 0
  reward = q[a] + noise
  return reward

# run TOTAL_STEPS rounds
for t in range(TOTAL_STEPS):
  # select action and get reward
  arm = epsilonGreedy(t)
  R = bandit(arm)
  # increase Nt
  Nt[arm] += 1
  # compute alpha
  alpha = 0.1 # (1 / Nt[arm])
  # update Q per reward as greedy bandits
  Q[arm] = Q[arm] +  alpha * (R - Q[arm])
  # save result for display

# plot results
optimalArm = np.argmax(q)
mostSelected = np.argmax(Nt)
optimalReward = q[optimalArm]
estimatedReward = Q[mostSelected]

print('Final action values:', q)
print('Selections:', Nt)
print('Optimal Arm:', optimalArm, ',', 'Selected:', mostSelected, 'for', Nt[mostSelected], 'times out of', TOTAL_STEPS)
print('Best Reward:', optimalReward, 'Estimated best reward:', estimatedReward)
