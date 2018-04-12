from enum import IntEnum
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


class TurtleActions(IntEnum):
    FD0 = 0
    FD1 = 1
    LT1 = 2
    RT1 = 3


class MnistTurtleEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    GRID_SIZE = 28

    def __init__(self):
        self.nD = 8  # number of directions
        # cell state 0: blank,  1: drawn black
        self.grid = np.asarray([[0]*self.GRID_SIZE]*self.GRID_SIZE, dtype=int)
        self.nA = len(TurtleActions)
        # row, col, direction, cell color(0 or 1)
        self.nS = self.GRID_SIZE * self.GRID_SIZE * self.nD * 2
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Start at bottom left, direction at 45 degrees, cell blank
        self.state = self._encode(self.GRID_SIZE-1, 0, 1, 0)
        return self.state

    def _get_next_cell(self, row, col, dirn):
        dr = [0, -1, -1, -1, 0, 1, 1, 1]
        dc = [1, 1, 0, -1, -1, -1, 0, 1]
        row_next = row + dr[dirn]
        col_next = col + dc[dirn]
        if (0 <= row_next < self.GRID_SIZE) and (0 <= col_next < self.GRID_SIZE):
            return row_next, col_next
        else:
            return row, col

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        row, col, direction, color = self._decode(self.state)
        if action == TurtleActions.FD0:
            row_next, col_next = self._get_next_cell(row, col, direction)
            color_next = self.grid[row_next, col_next]
            dir_next = direction
        elif action == TurtleActions.FD1:
            row_next, col_next = self._get_next_cell(row, col, direction)
            color_next = 1
            dir_next = direction
            self.grid[row_next, col_next] = 1
        elif action == TurtleActions.LT1:
            row_next, col_next = row, col
            color_next = color
            dir_next = (direction - 1 + self.nD) % self.nD
        elif action == TurtleActions.RT1:
            row_next, col_next = row, col
            color_next = color
            dir_next = (direction + 1) % self.nD

        state_next = self._encode(row_next, col_next, dir_next, color_next)
        reward = None
        done = False
        return state_next, reward, done, {}

    def _encode(self, row, col, direction, color):
        s = row
        for a, b in ((self.GRID_SIZE, col), (self.nD, direction), (2, color)):
            s *= a
            s += b
        return s

    def _decode(self, state):
        out = []
        for a in (2, self.nD, self.GRID_SIZE):
            out.append(state % a)
            state = state // a
        out.append(state)
        assert 0 <= state < self.GRID_SIZE
        return reversed(out)

    def get_grid_bitmap(self):
        ''' We don't need to do anything here. MNIST images also have same
            format: [0,255] values for pixels which are actually scaled to [0,1]
            before training models. We already have values 0 or 1 for cell color
        '''
        return np.array(self.grid, dtype=np.float)

    def render(self, mode='human'):
        if mode == 'human':
            plt.imshow(self.grid, cmap='gray_r')
            plt.show()
