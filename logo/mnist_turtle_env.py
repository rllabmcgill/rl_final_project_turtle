from enum import IntEnum
import gym
import numpy as np
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
    STOP = 4


class MnistTurtleEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    GRID_SIZE = 28

    def __init__(self, digit):
        self.digit = digit
        self.nD = 8  # number of directions
        # cell state 0: blank,  1: drawn black
        self.grid = np.asarray([[0]*self.GRID_SIZE]*self.GRID_SIZE, dtype=int)
        self.nA = len(TurtleActions)
        # row, col, direction, cell color(0 or 1)
        self.nS = self.GRID_SIZE * self.GRID_SIZE * self.nD * 2
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.GRID_SIZE, self.GRID_SIZE, 1))
        self.row = self.col = self.direction = 0

        # mnist classifier
        model_raw, _, _ = selector.select('mnist', cuda=False)
        self.mnist_model = model_raw
        self.preprocess = transforms.Compose([
                               transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, row=None, col=None, direction=None, color=None):
        row = row or self.row
        col = col or self.col
        direction = direction or self.direction
        color = color or self.grid[row, col]
        self.row, self.col, self.direction, self.grid[row][col] = row, col, direction, color
        self.turtle_state = self._encode(row, col, direction, color)

    def reset(self):
        self.set_state(row=np.random.randint(0, self.GRID_SIZE),
                       col=np.random.randint(0, self.GRID_SIZE),
                       direction=np.random.randint(0, 8),
                       color=0)
        return np.expand_dims(self.get_grid_bitmap(),axis=2)

    @property
    def turtle_pos(self): return self.row, self.col, self.direction

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

        reward = 0.0
        done = False
        if action == TurtleActions.FD0:
            row_next, col_next = self._get_next_cell(self.row, self.col, self.direction)
            self.set_state(row=row_next, col=col_next)
        elif action == TurtleActions.FD1:
            row_next, col_next = self._get_next_cell(self.row, self.col, self.direction)
            self.set_state(row=row_next, col=col_next, color=1)
        elif action == TurtleActions.RT1:
            self.set_state(direction=(self.direction - 1 + self.nD) % self.nD)
        elif action == TurtleActions.LT1:
            self.set_state(direction=(self.direction + 1) % self.nD)
        elif action == TurtleActions.STOP:
            reward, _ = self.calc_reward()
            done = True

        #reward, done = self.calc_reward()
        return np.expand_dims(self.get_grid_bitmap(),axis=2), reward, done, {'turtle_state': self.turtle_state}

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

    def calc_reward(self):
        '''
        Calculate reward from MNIST image for a given digit
        :return: Reward, done
        '''
        bitmap = self.get_grid_bitmap()
        bitmap = self.preprocess(torch.from_numpy(bitmap[np.newaxis,:]).float()) # 1x28x28
        data = Variable(bitmap, requires_grad=False)
        data = data.unsqueeze(0) # 1x1x28x28
        logits = self.mnist_model(data)
        prob = F.softmax(logits,dim=1)
        p_digit = prob[0][self.digit].data[0]

        done = any(prob[0][d].data[0] > 0.95 for d in range(10))
        return p_digit, done

    def render(self, mode='human',close=False):
        print('Close: ', close)
        if mode == 'human' and not close:
            plt.imshow(self.grid, cmap='gray_r')
            plt.show()
