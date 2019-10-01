import numpy as np
import numpy.random as npr

"""
    Bandit with 10 arms that can select an action at a given time step
    and update internal action-values and estimated action-values
"""
class Bandit:

    numArms = 10

    def __init__(self, steps):
        self.e = 0.1
        self.alpha = 0.1
        self.reward = 0
        self.actIdx = 0
        self.optimal = False
        self.armCount = [0] * Bandit.numArms

        # expected value for actions
        # self.q = np.array([npr.normal(0, 0.1) for i in range(10)])
        self.q = np.zeros(Bandit.numArms)
        # estimated value for actions
        self.Q = np.zeros(Bandit.numArms)

        # count of optimal actions for each time step
        self.opActions = np.zeros(steps)


    """
        Selects action using e-greedy method and updates state of bandit to
        include reward returned from action, index of action, and a flag if the optimal action was taken. 
    """
    def chooseAction(self):
        explore = npr.uniform(0, 1)
        exploit = 1 - self.e

        self.optimal = False

        if explore < exploit:
            # select action with largest estimated value
            self.actIdx = np.argmax(self.Q, axis=0)

        else:
            # select random action
            self.actIdx = npr.choice(Bandit.numArms)

        self.armCount[self.actIdx] += 1

        # determine if optimal action was taken
        if self.actIdx == np.argmax(self.q, axis=0):
            self.optimal = True

        # calculate reward
        self.reward = npr.normal(self.q[self.actIdx], 0.1)

    """
       Updates estimated action values and expected action values
       with the reward retrieved from selected action.
       
       :param step - current step number in iteration
       :param stepSize - False if sample average action value method is used.
                         True if step size parameter action value method is used
    """
    def takeStep(self, step, stepSize=False):

        # choose action
        self.chooseAction()

        i = self.actIdx

        # update estimated values for action
        n = self.armCount[i]

        scale = (1 / n) if not stepSize else self.alpha
        self.Q[i] += scale * (self.reward - self.Q[i])

        # update action values
        noise = npr.normal(0, 0.01)
        self.q += noise

        if self.optimal:
            self.opActions[step - 1] += 1

        return self.reward

    def getOpActions(self):
        return self.opActions
