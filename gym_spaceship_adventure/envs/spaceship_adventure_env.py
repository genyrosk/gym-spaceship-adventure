import copy
import sys

import gym
from gym import spaces, utils
from gym.utils import seeding
import numpy as np

#
# Possible actions
#
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
BOOST_LEFT = 4
BOOST_DOWN = 5
BOOST_RIGHT = 6
BOOST_UP = 7

BOOST_TO_DIRECTION = {
    BOOST_LEFT: LEFT,
    BOOST_DOWN: DOWN,
    BOOST_RIGHT: RIGHT,
    BOOST_UP: UP,
}

REGULAR_ACTIONS = [
    LEFT,
    DOWN,
    RIGHT,
    UP,
]

BOOST_ACTIONS = [
    BOOST_LEFT,
    BOOST_DOWN,
    BOOST_RIGHT,
    BOOST_UP,
]

ACTIONS = REGULAR_ACTIONS + BOOST_ACTIONS

#
# Nomenclature:
# 'S' = starting point
# 'G' = goal point
# 'A' = asteroid
# '-' = empty space
#
DEFAULT_MAP = [
    "AA-A-G",
    "--A---",
    "----AA",
    "--A--A",
    "A--A--",
    "S-----",
]


class IllegalAction(Exception):
    pass


def categorical_sample(category_probs, np_random):
    """
    Sample from categorical distribution `category_probs`
    Each row specifies class probabilities
    """
    probs_n = np.asarray(category_probs)
    cumsum_probs_n = np.cumsum(probs_n)
    return (cumsum_probs_n > np_random.rand()).argmax()


def generate_random_map(size=6, p=0.7):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is empty
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        start_row, start_col = (size - 1, 0)
        frontier.append((start_row, start_col))  # frontier tracks the path we're taking
        # if frontier[] becomes empty then we have exhausted all valid paths
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for dr, dc in directions:
                    r_new = r + dr
                    c_new = c + dc
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "A":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        result = np.random.choice(["-", "A"], (size, size), p=[p, 1 - p])
        result[-1][0] = "S"  # set starting point
        result[0][-1] = "G"  # set goal point
        valid = is_valid(result)
    return ["".join(x) for x in result]


class SpaceshipAdventureEnv(gym.Env):
    """
    The surface is grid_mapribed using a grid like the following
        AA-A-G
        --A---
        ----AA
        --A--A
        A--A--
        S-----
    S : starting point, safe
    - : empty space, safe
    A : asteroid, colliding means game over
    G : goal
    The episode ends when you reach the goal or collide with an asteroid.
    You receive an award of -1 for each move, 10 for reaching the goal,
    and zero otherwise.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, use_default_map=True, size=6, p=0.7, max_boosts=float("inf"), action_randomness=0.0):
        """
        :param use_default_map
        :param size: size of each side of the grid
        :param p: probability that a tile is empty
        :param max_boosts: maximum number of booosts the spaceship can use
        :param action_randomness: action uncertainty

        Every environment should be derived from gym.Env and at
        least contain the variables observation_space and action_space
        specifying the type of possible observations and actions
        using spaces.Box or spaces.Discrete.

        Has the following members
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)
        (*) dictionary of lists, where
            P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS
        """
        if use_default_map:
            grid_map = DEFAULT_MAP
        else:
            grid_map = generate_random_map(size=size, p=p)

        self.grid_map = grid_map = np.asarray(grid_map, dtype="c")
        self.nrow, self.ncol = numRows, numCols = grid_map.shape
        numActions = 8
        numStates = numRows * numCols
        self.nA = numActions
        self.nS = numStates
        self.last_action = None  # for rendering
        self.max_boosts = max_boosts
        self.boosts_used = 0
        self.cumulative_reward = 0

        # Expose API
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # Initial state distribution
        isd = np.array(grid_map == b"S").astype("float64").ravel()
        isd /= isd.sum()
        self.isd = isd

        # Transition matrix
        # P_mat[state_idx][action_idx] = (probability, state_index, reward, done)
        P_mat = {
            state_idx: {action: [] for action in range(self.nA)}
            for state_idx in range(self.nS)
        }
        self.P_mat = P_mat

        def coords_to_flat_idx(row, col):
            return row * numRows + col

        def next_square(row, col, action):
            if action == LEFT:
                col = max(col - 1, 0)
            elif action == DOWN:
                row = min(row + 1, numRows - 1)
            elif action == RIGHT:
                col = min(col + 1, numCols - 1)
            elif action == UP:
                row = max(row - 1, 0)
            return (row, col)

        def next_state(prev_row, prev_col, action):
            # copy prev state
            # breakpoint()
            row = copy.copy(prev_row)
            col = copy.copy(prev_col)

            # calculate the path
            if action in REGULAR_ACTIONS:
                path = [next_square(row, col, action)]
            elif action in BOOST_ACTIONS:
                action_direction = BOOST_TO_DIRECTION[action]
                path = []
                for _ in range(4):
                    (row, col) = next_square(row, col, action_direction)
                    path.append((row, col))

            # for given path: get new state, reward and done
            new_state_idx = coords_to_flat_idx(prev_row, prev_col)
            reward = -1
            done = False
            for square in path:
                r, c = square
                new_state_idx = coords_to_flat_idx(r, c)
                letter = grid_map[r, c]
                if bytes(letter) == b"G":
                    done = True
                    reward = 10
                    break
                elif bytes(letter) == b"A":
                    done = True
                    break

            return (new_state_idx, reward, done)

        #
        # The map layout is deterministic
        # └-> pre-calculate all possible states:
        # ... for each possible location where the ship could be located...
        for row in range(numRows):
            for col in range(numCols):
                state_idx = coords_to_flat_idx(row, col)
                # ... and for every possible action ...
                for action in ACTIONS:
                    # ... there is a known transition
                    #
                    # Note:
                    #   typically we talk about a number of possible transitions,
                    #   each defined by:
                    #    - a probability of it ocurring
                    #    - a new state
                    #    - a reward
                    #    - whether the simulation is over or not (i.e. `done`)
                    #   In our scenario, we only have one possible transition for a
                    #   given action since it's deterministic, thus its probability is 1.0
                    transitions = []
                    letter = grid_map[row, col]
                    if letter in b"GA":
                        # simulation complete, just mark as done
                        prob = 1.0
                        reward = 0
                        done = True
                        transitions.append((prob, state_idx, reward, done))
                    else:
                        # Boosts are deterministic, while regular actions can have some randomness
                        if action_randomness and action in REGULAR_ACTIONS:
                            # breakpoint()
                            for _action in [(action - 1) % 4, action, (action + 1) % 4]:
                                if _action == action:
                                    prob = 1.0 - action_randomness
                                else:
                                    prob = action_randomness / 2.0
                                new_state_idx, reward, done = next_state(row, col, _action)
                                transitions.append((prob, new_state_idx, reward, done))
                        else:
                            prob = 1.0
                            new_state_idx, reward, done = next_state(row, col, action)
                            transitions.append((prob, new_state_idx, reward, done))
                    # update the transition matrix
                    P_mat[state_idx][action] = transitions

        # set the seed and the initial state
        self.seed()
        self.state = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_possible_actions(self):
        actions = REGULAR_ACTIONS
        if not self.boosts_used > self.max_boosts:
            actions += BOOST_ACTIONS
        return actions

    def action_is_valid(self, action):
        return action in self.get_possible_actions()

    def step(self, action):
        """
        This method is the primary interface between environment and agent.
        Paramters:
            action: int
                    the index of the respective action (if action space is discrete)
        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        if not self.action_is_valid(action):
            raise IllegalAction(f"action {action} is not possible")

        if action in BOOST_ACTIONS:
            self.boosts_used += 1

        transitions = self.P_mat[self.state][action]
        idx = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, state_idx, reward, done = transitions[idx]
        self.state = state_idx
        self.last_action = action
        self.cumulative_reward += reward
        info = {"prob": prob}
        return (int(self.state), reward, done, info)

    def reset(self):
        """
        This method resets the environment to its initial values.

        Returns:
          observation:  array
                        the initial state of the environment
        """
        self.state = categorical_sample(self.isd, self.np_random)
        self.last_action = None
        self.cumulative_reward = 0
        return int(self.state)

    def render(self, mode="human"):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.state // self.ncol, self.state % self.ncol
        grid_map = self.grid_map.tolist()
        grid_map = [[char.decode("utf-8") for char in line] for line in grid_map]
        grid_map[row][col] = utils.colorize(grid_map[row][col], "red", highlight=True)
        if self.last_action is not None:
            last_action = [
                "Left",
                "Down",
                "Right",
                "Up",
                "Boost_Left",
                "Boost_Down",
                "Boost_Right",
                "Boost_Up",
            ][self.last_action]
            outfile.write(f"  ({last_action})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in grid_map) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
