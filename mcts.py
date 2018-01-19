from __future__ import print_function

import random
import utils

NODE_COUNTER = 0

class MCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, tree_policy, default_policy, backup):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup
        self.node_counter = 0

    def __call__(self, root, n=100):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(n):
            (node,count) = _get_next_node(root, self.tree_policy)
            node.reward = self.default_policy(node)
            self.node_counter+=count
            self.backup(node)
        # print ()
        return (utils.rand_max(root.children.values(), key=lambda x: x.q).action,self.node_counter)


def _expand(state_node):
    action = random.choice(state_node.untried_actions)
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy):
    best_action_node = utils.rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy):
    node_counter = 0
    while not state_node.state.is_terminal():
        node_counter+=1
        if state_node.untried_actions:
            return (_expand(state_node),node_counter)
        else:
            state_node = _best_child(state_node, tree_policy)

    return (state_node,node_counter)
