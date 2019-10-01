
"""
:author Isaac Buitrago

Design and conduct an experiment to demonstrate the difficulties that sample-average methods have

Test
(1) an action-value method using sample averages, with incrementally computed and q estimates that all start from 0.
(2) an action-value method using a constant step-size parameter, alpha = 0.1, epsilon=0.1, and iterate 10,000 steps for each bandit.

Repeat both methods for 500 independent runs, and report
(1) average rewards
(2) the ratio of optimal actions averaged over those 500 runs
"""

import numpy as np
import sys
from Bandit import Bandit

# repeat for 500 runs
# 2. Return a reward for arm i sampled from a normal distribution around true value
# 3. update estimate for arm that was pulled
# 4. update true values with noise

# select arm based on epsilon greedy, arm with largest Q value

# optimal value is the action with max q* value
# if lever being pulled has max q*, optimal action was taken



def main(args):

    steps = 10000
    runs = 300

    # average rewards over time steps
    avgReward1 = np.zeros(steps)
    avgReward2 = np.zeros(steps)

    # percent of optimal actions
    opActions1 = np.zeros(steps)
    opActions2 = np.zeros(steps)

    # validate command line args
    if len(args) != 2:
        print("Error: Output file not provided")
        print("Usage: driver.py result.out")
        exit()

    """
        Bandits using sample averages
    """
    for r in range(runs):
        bandit = Bandit(steps)

        for step in range(1, steps + 1):
            reward = bandit.takeStep(step)
            avgReward1[step - 1] += reward

        opActions1 += bandit.getOpActions()

    opActions1 /= runs
    avgReward1 /= runs


    """
        Bandits using step size parameters
    """
    for r in range(runs):
        bandit = Bandit(steps)

        for step in range(1, steps + 1):
            reward = bandit.takeStep(step, stepSize=True)
            avgReward2[step - 1] += reward

        opActions2 += bandit.getOpActions()

    opActions2 /= runs
    avgReward2 /= runs

    # save data in file
    try:

        fn = args[1]
        file = open(fn, 'w')
        np.savetxt(file, (avgReward1, opActions1), newline="\n")
        np.savetxt(file, (avgReward2, opActions2), newline="\n")

    except FileNotFoundError as e:
        print(f"Error: {e.strerror}")

    finally:
        file.close()


# start algorithm
main(sys.argv)