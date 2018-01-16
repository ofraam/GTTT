import numpy as np
# import mcts
# import mcts_local
from mcts import *
from tree_policies import *
from default_policies import *
from backups import *
from utils import *
from graph import *
import random

class MazeAction(object):
    def __init__(self, move):
        self.move = np.asarray(move)

    def __eq__(self, other):
        return all(self.move == other.move)

    def __hash__(self):
        return 10*self.move[0] + self.move[1]

class MazeState(object):
    def __init__(self, pos):
        self.pos = np.asarray(pos)
        self.actions = [MazeAction([1, 0]),
                        MazeAction([0, 1]),
                        MazeAction([-1, 0]),
                        MazeAction([0, -1])]

    def perform(self, action):
        pos = self.pos + action.move
        pos = np.clip(pos, 0, 2)
        return MazeState(pos)

    def reward(self, parent, action):
        if all(self.pos == np.array([2, 2])):
            return 10
        else:
            return -1

    def is_terminal(self):
        # # return True
        # if (random.random()<0.05):
        return False

    def __eq__(self, other):
        return all(self.pos == other.pos)

    def __hash__(self):
        return 10 * self.pos[0] + self.pos[1]

if __name__ == "__main__":
    mcts = MCTS(tree_policy=UCB1(c=1.41),
                default_policy=immediate_reward,
                backup=monte_carlo)

    root = StateNode(None, MazeState([0, 0]))
    best_action = mcts(root)
    print best_action.move