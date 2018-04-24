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


class Marker(IntEnum):
    Empty = 0
    Target = 1
    Drawn = 2
    Connected = 3


class Colors:
    Black = 0.0
    White = 1.0
    Green = [0.0, 1.0, 0.0]
    Red = [1.0, 0.0, 0.0]
    Blue = [0.0, 0.0, 1.0]
    Yellow = [1.0, 1.0, 0.0]
    Gray = [0.5, 0.5, 0.5]
    Cyan = [0.0, 1.0, 1.0]


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
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int)
        self.rgb_grid = np.ones((self.GRID_SIZE, self.GRID_SIZE, 3), dtype=np.float)
        self.nA = len(TurtleActions)

        self.nS = self.GRID_SIZE * self.GRID_SIZE
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.row = self.col = self.direction = 0
        self.turtle_colors = [Colors.Blue, Colors.Green, Colors.Yellow, Colors.Red]

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, pos=None, direction=None, draw=None):
        if pos is not None:
            self.update_rgb_grid()
            self.row, self.col = pos
        if direction is not None:
            self.direction = direction
        if draw is True:
            if self.grid[self.row, self.col] == Marker.Empty:
                self.grid[self.row, self.col] = Marker.Drawn
        self.rgb_grid[self.row, self.col] = self.turtle_colors[self.direction]

    def update_rgb_grid(self):
        #for d in range(self.nD):
        #    r, c = self._get_next_cell(self.row, self.col, d)
        r, c = self.row, self.col
        marker = self.grid[r, c]
        if marker == Marker.Empty:
            color = Colors.White
        elif marker in (Marker.Target, Marker.Connected):
            color = Colors.Black
        elif marker == Marker.Drawn:
            color = Colors.Cyan
        self.rgb_grid[r, c] = color

    def reset(self):
        self.grid.fill(Marker.Empty)
        self.rgb_grid.fill(Colors.White)
        self.total_connected = 0
        for r, c in self.target_dots:
            self.grid[r, c] = Marker.Target
            self.rgb_grid[r, c] = Colors.Black

        self.set_state(pos=random.sample(self.target_dots, 1)[0],
                       direction=np.random.randint(0, self.nD))
        return np.expand_dims(self.get_grid_bitmap(), axis=2)

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
            self.set_state(pos=(row_next,col_next))
        elif action == TurtleActions.FD1:
            row_next, col_next = self._get_next_cell(self.row, self.col, self.direction)
            self.set_state(pos=(row_next,col_next), draw=True)
        elif action == TurtleActions.RT1:
            self.set_state(direction=(self.direction - 1 + self.nD) % self.nD)
        elif action == TurtleActions.LT1:
            self.set_state(direction=(self.direction + 1) % self.nD)

        reward, done = self.calc_reward()
        return np.expand_dims(self.get_grid_bitmap(), axis=2), reward, done, {}

    def get_grid_bitmap(self):
        return np.array(self.rgb_grid, dtype=np.float)

    def is_connected(self, p1, p2):
        tpos = (self.row, self.col)
        if tpos != p1 and tpos != p2:
            return False
        p1, p2 = (p1, p2) if (p1 < p2) else (p2, p1)
        if p1[0] == p2[0]:
            return all((self.grid[p1[0], c] == Marker.Drawn) for c in range(p1[1]+1, p2[1]))
        if p1[1] == p2[1]:
            return all((self.grid[r, p1[1]] == Marker.Drawn) for r in range(p1[0]+1, p2[0]))

    def mark_connected(self, p1, p2):
        p1, p2 = (p1, p2) if (p1 < p2) else (p2, p1)
        if p1[0] == p2[0]:
            for c in range(p1[1]+1, p2[1]):
                self.grid[p1[0], c] = Marker.Connected
                self.rgb_grid[p1[0], c] = Colors.Black
        if p1[1] == p2[1]:
            for r in range(p1[0]+1, p2[0]):
                self.grid[r, p1[1]] = Marker.Connected
                self.rgb_grid[r, p1[1]] = Colors.Black

    def calc_reward(self):
        reward = -1.0
        for p1, p2 in self.connections:
            if self.is_connected(p1, p2):
                self.total_connected += 1
                reward = 1.0
                self.mark_connected(p1, p2)
                break
        done = (self.total_connected == len(self.connections))
        return reward, done

    def render(self, mode='human',close=False):
        print('Close: ', close)
        if mode == 'human' and not close:
            plt.imshow(self.rgb_grid)
            plt.show()
