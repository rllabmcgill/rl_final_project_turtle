from enum import IntEnum
import gym
import numpy as np, random
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from utee import selector
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


class TurtleActions(IntEnum):
    FD0 = 0
    FD1 = 1
    LT1 = 2
    RT1 = 3

all_connections = {
    3: {((6,6),(6,11)),  # Top lines
        ((6,11),(6,16)),
        ((6,16),(6,21)),
        ((6,21),(10,21)),  # Right lines
        ((10,21),(13,21)),
        ((13,21),(17,21)),
        ((17,21),(21,21)),
        ((21,21),(21,16)),  # Bottom lines
        ((21,16),(21,11)),
        ((21,11),(21,6)),
        ((13,11),(13,16)),  # Middle Lines
        ((13,16),(13,21))
    }
}

class ConnectDotsEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    GRID_SIZE = 28

    def __init__(self, digit):
        self.digit = digit
        self.connections = all_connections[digit]
        self.target_dots = set()
        for p1, p2 in self.connections:
            self.target_dots.add(p1)
            self.target_dots.add(p2)
        self.nD = 4  # number of directions
        # cell state 0: blank,  1: target dots, 0.5 turtle drawing
        self.grid = np.asarray([[0]*self.GRID_SIZE]*self.GRID_SIZE, dtype=np.float)
        self.nA = len(TurtleActions)

        self.nS = self.GRID_SIZE * self.GRID_SIZE
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.row = self.col = self.direction = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, row=None, col=None, direction=None, color=None):
        if row is not None:
            self.row = row
        if col is not None:
            self.col = col
        if direction is not None:
            self.direction = direction
        if color is not None:
            if (self.row, self.col) in self.target_dots:
                color = 1.0
            if self.grid[self.row][self.col] < 0.9:
                self.grid[self.row][self.col] = color

    def reset(self):
        self.grid.fill(0)
        self.total_connected = 0
        for r, c in self.target_dots:
            self.grid[r, c] = 1.0

        start = random.sample(self.target_dots, 1)[0]
        self.set_state(row=start[0],
                       col=start[1],
                       direction=np.random.randint(0, self.nD),
                      )
        return np.expand_dims(self.get_grid_bitmap(),axis=2)

    @property
    def turtle_pos(self): return self.row, self.col, self.direction

    def _get_next_cell(self, row, col, dirn):
        dr = [0, -1, 0, 1]
        dc = [1, 0, -1, 0]
        row_next = row + dr[dirn]
        col_next = col + dc[dirn]
        if (0 <= row_next < self.GRID_SIZE) and (0 <= col_next < self.GRID_SIZE):
            return row_next, col_next
        else:
            return row, col

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if action == TurtleActions.FD0:
            row_next, col_next = self._get_next_cell(self.row, self.col, self.direction)
            self.set_state(row=row_next, col=col_next)
        elif action == TurtleActions.FD1:
            row_next, col_next = self._get_next_cell(self.row, self.col, self.direction)
            self.set_state(row=row_next, col=col_next, color=0.5)
        elif action == TurtleActions.RT1:
            self.set_state(direction=(self.direction - 1 + self.nD) % self.nD)
        elif action == TurtleActions.LT1:
            self.set_state(direction=(self.direction + 1) % self.nD)

        reward, done = self.calc_reward()
        return np.expand_dims(self.get_grid_bitmap(),axis=2), reward, done, {}

    def get_grid_bitmap(self):
        return np.array(self.grid, dtype=np.float)

    def is_connected(self, p1, p2):
        tpos = (self.row, self.col)
        if tpos != p1 and tpos != p2:
            return False
        p1, p2 = (p1, p2) if (p1 < p2) else (p2, p1)
        if p1[0] == p2[0]:
            return all((0.1 < self.grid[p1[0], c] < 0.9) for c in range(p1[1]+1, p2[1]))
        if p1[1] == p2[1]:
            return all((0.1 < self.grid[r, p1[1]] < 0.9) for r in range(p1[0]+1, p2[0]))

    def paint_black(self, p1, p2):
        p1, p2 = (p1, p2) if (p1 < p2) else (p2, p1)
        if p1[0] == p2[0]:
            for c in range(p1[1]+1, p2[1]):
                self.grid[p1[0], c] = 1.0
        if p1[1] == p2[1]:
            for r in range(p1[0]+1, p2[0]):
                self.grid[r, p1[1]] = 1.0

    def calc_reward(self):
        reward = -1.0
        for p1, p2 in self.connections:
            if self.is_connected(p1, p2):
                self.total_connected += 1
                reward = 1.0
                self.paint_black(p1, p2)
                break
        done = (self.total_connected == len(self.connections))
        return reward, done

    def render(self, mode='human',close=False):
        print('Close: ', close)
        if mode == 'human' and not close:
            plt.imshow(self.grid, cmap='gray_r')
            plt.show()
