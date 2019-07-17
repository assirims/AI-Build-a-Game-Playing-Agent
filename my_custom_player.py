# Crdit:
# https://github.com/udacity/artificial-intelligence
# https://github.com/zhengrui315/Udacity-AIND-Adversarial-Search
# https://github.com/rimoune/AIND-Build_a_Game_Playing_Agent
# https://github.com/asvcode/AIND---Build-a-game-playing-agent
# https://github.com/LeanderLXZ/isolation-agent

from sample_players import DataPlayer
import random
import math
import copy
from isolation.isolation import _WIDTH, _HEIGHT


class CustomPlayer_MiniMax(DataPlayer):
    """Implement customized agent to play knight's Isolation"""

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        if state.ply_count < 4:
            if state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                best_move = alpha_beta_search(state, self.player_id, depth)
            self.queue.put(best_move)


CustomPlayer = CustomPlayer_MiniMax


class CustomPlayer_MCTS(DataPlayer):
    """An player agent via Monte Carlo Tree Search"""

    def mcts(self, state):
        root = MCTS_Node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(iter_limit):
            child = tree_policy(root)
            if not child:
                continue
            reward = default_policy(child.state)
            backup(child, reward)

        idx = root.children.index(best_child(root))
        return root.children_actions[idx]

    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))


class MCTS_Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent

    def add_child(self, child_state, action):
        child = MCTS_Node(child_state, self)
        self.children.append(child)
        self.children_actions.append(action)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def full_search(self):
        return len(self.children_actions) == len(self.state.actions())


FACTOR = 1.0
iter_limit = 20


def tree_policy(node):
    """Return status of child node as not searched or searched with best score."""
    while not node.state.terminal_test():
        if not node.full_search():
            return explore(node)
        node = best_child(node)
    return node


def explore(node):
    implmented_actions = node.children_actions
    possible_actions = node.state.actions()
    for action in possible_actions:
        if action not in implmented_actions:
            new_state = node.state.result(action)
            node.add_child(new_state, action)
            return node.children[-1]


def best_child(node):
    """Return both of the child node and best score"""
    best_score = float("-inf")
    best_children = []
    for child in node.children:
        exploit = child.reward / child.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
        score = exploit + FACTOR * explore
        if score == best_score:
            best_children.append(child)
        elif score > best_score:
            best_children = [child]
            best_score = score
    if len(best_children) == 0:
        return None
    return random.choice(best_children)


def default_policy(state):
    """Return results of random search"""
    init_state = copy.deepcopy(state)
    while not state.terminal_test():
        action = random.choice(state.actions())
        state = state.result(action)

    return -1 if state._has_liberties(init_state.player()) else 1


def backup(node, reward):
    """Update nodes' path information"""
    while node != None:
        node.update(reward)
        node = node.parent
        reward *= -1


def alpha_beta_search(state, play_id, depth=3):
    def min_value(state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(play_id)
        if depth <= 0:
            return score(state, play_id)
        value = float("inf")
        for action in state.actions():
            value = min(value, max_value(state.result(action), alpha, beta, depth-1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(state, alpha, beta, depth):
        if state.terminal_test():
            return state.utility(play_id)
        if depth <= 0:
            return score(state, play_id)
        value = float("-inf")
        for action in state.actions():
            value = max(value, min_value(state.result(action), alpha, beta, depth-1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    alpha = float("-inf")
    beta = float("inf")
    best_score = float("-inf")
    best_move = None
    for action in state.actions():
        value = min_value(state.result(action), alpha, beta, depth-1)
        alpha = max(alpha, value)
        if value >= best_score:
            best_score = value
            best_move = action
    return best_move


def distance(state):
    own_loc = state.locs[state.ply_count % 2]
    x_player, y_player = own_loc // (_WIDTH + 2), own_loc % (_WIDTH + 2)

    return min(x_player, _WIDTH + 1 - x_player, y_player, _HEIGHT - 1 - y_player)


def score(state, play_id):
    own_loc = state.locs[play_id]
    opp_loc = state.locs[1 - play_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)

    dis = distance(state)
    if dis >= 2:
        return 2*len(own_liberties) - len(opp_liberties)
    else:
        return len(own_liberties) - len(opp_liberties)
