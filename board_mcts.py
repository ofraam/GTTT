import numpy as np
# import mcts
# import mcts_local
from mcts import *
from tree_policies import *
from default_policies import *
from backups import *
from utils import *
from graph import *
import config as c
import os
from computational_model import *
import copy
import random


class TicTacToeAction(object):
    def __init__(self, move):
        self.move = move

    def __eq__(self, other):
        return self.move == other.move

    def __hash__(self):
        return self.move


class TicTactToeState(object):
    def __init__(self, board, player, depth):
        self.board = board
        self.actions = self.board.get_free_spaces();
        self.player = player
        self.depth = depth

    def perform(self, action):
        new_board = copy.deepcopy(self.board)
        if action not in new_board.get_free_spaces():
            print 'prob'
        new_board.add_marker(action, player=self.player)

        if self.player == c.HUMAN:
            player = c.COMPUTER
        else:
            player=c.HUMAN
        depth = self.depth+1
        return TicTactToeState(new_board, player, depth)

    def reward(self, parent, action):
        # return -1*self.board.obj_interaction(self.player)
        if self.board.is_terminal():
            if self.board.obj(self.player) == c.LOSE_SCORE:
                return 1
            # if self.board.obj(self.player) == c.WIN_SCORE:
            else:
                return -1
        elif self.board.check_possible_win(math.ceil(self.depth/2.0)):
            return 0
        else:
            return -1
        return 0

    def is_terminal(self):
        if self.board.is_terminal():
            return True
        if c.WIN_DEPTH == self.depth:
            return True
        # # return True
        # if (random.random()<0.05):
        return False

    def __eq__(self, other):
        # return (str(self.board) == str(other.board)) & (str(self.player) == str(other.player)) & (str(self.depth) == str(other.depth))
        if (len(self.actions)!=len(other.actions)):
            return False
        for act in self.actions:
            if act not in other.actions:
                return False
        return True

    def __hash__(self):
        # print self.board.board
        # print hash(str(self.board.board)+str(self.player)+'_'+str(self.depth))
        return hash(str(self.board.board)+str(self.player)+'_'+str(self.depth))


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

    for filename in os.listdir("predefinedBoards/"):
        # print filename
        if filename.startswith("6"):
            file_path = "examples/board_6_4.txt"
            # continue
            # if not(filename.startswith("6by6_easy")):
            #     continue

        else:
            # if filename.startswith("10by10_easy"):
            # if not(filename.startswith("10by10_easy")):
            #   continue
            file_path = "examples/board_10_5.txt"
            # continue
        chosen_moves = {}
        num_runs = 50
        num_correct = 0.0
        print filename
        total_nodes= 0.0
        for i in range(0,num_runs):
            game = start_game(file_path)

            win_depth = fill_board_from_file("predefinedBoards/"+filename,game)
            c.WIN_DEPTH = win_depth
            root = StateNode(None,TicTactToeState(game.board,game.whos_turn,0))

            best_action, num_nodes = mcts(root, n=40)


            if best_action in chosen_moves.keys():
                chosen_moves[best_action] += 1
            else:
                chosen_moves[best_action] = 1

            if best_action in c.WIN_MOVES:
                num_correct += 1
            total_nodes += num_nodes

        print num_nodes/num_runs
        print chosen_moves
        print num_correct/num_runs





    #
    # root = StateNode(None, MazeState([0, 0]))
    # best_action = mcts(root)
    # print best_action.move