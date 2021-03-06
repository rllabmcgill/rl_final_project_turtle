import sys, os, logging
sys.path.append('%s'%os.path.abspath('/home/ndg/users/dparek/rl_final_project_turtle'))
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten  # , MaxPooling2D
from keras.optimizers import Adam
import numpy as np
import random
import pickle
from collections import deque
from keras import backend as K
from logo.connect_dots_env import ConnectDotsEnv


def get_logger(stream=sys.stdout, name=''):
    logging.basicConfig(stream=stream, level=logging.INFO,
                        format='%(asctime)s | %(threadName)10s | %(levelname)s | ' +
                        '%(module)s:%(lineno)d | %(message)s')
    return logging.getLogger(name)

log = get_logger()


class DQNAgent:
    def __init__(self, action_size, input_shape):
        self.action_size = action_size
        self.input_shape = input_shape
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_final = 0.0001
        self.epsilon_init = 1.0
        self.observe_until = 50
        self.explore_until = 50000
        self.learning_rate = 0.001
        self.update_target_every = 100
        self.model = self._build_cnn()
        self.target_model = self._build_cnn()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_cnn(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(5, 5), strides=2, activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.20))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size, i_episode):
        if self.epsilon > self.epsilon_final and i_episode > self.observe_until:
            self.epsilon -= (self.epsilon_init - self.epsilon_final) / self.explore_until

        if i_episode % self.update_target_every == 0:
            self.update_target_model()
        if i_episode < self.observe_until:
            return

        num_samples = min(len(self.memory), batch_size)
        minibatch = random.sample(self.memory, num_samples)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train_agent(env, num_episodes=1000, batch_size=32, steps_per_episode=500):
    action_size = env.action_space.n
    input_shape = (env.GRID_SIZE, env.GRID_SIZE, 3)
    agent = DQNAgent(action_size, input_shape)
    # agent.load("./trained_models/logo_mnist_ddqn.h5")
    all_rewards, all_timesteps, all_connected = [], [], []
    best_grids = [0, -1e8, []]
    grids_11 = [0, -1e8, []]
    grids_12 = [0, -1e8, []]

    for i_episode in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        t_grids, t_reward = [], 0.0
        for t_step in range(steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)
            t_grids.append(state)
            t_reward += reward
            state = next_state
            if done:
                break

        all_rewards.append(t_reward)
        all_timesteps.append(t_step)
        all_connected.append(env.total_connected)

        if t_reward > best_grids[1]:
            best_grids = (i_episode, t_reward, t_grids)

        if env.total_connected == 11:
            grids_11 = (i_episode, t_reward, t_grids)

        if env.total_connected == 12:
            grids_12 = (i_episode, t_reward, t_grids)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size, i_episode)

        if i_episode >= num_episodes-1 or i_episode % 5000 == 0:
            with open('./saved_stats/connect_dots_dqn_gpu_grids_preferdots_%s.pkl' % i_episode, 'wb') as fout:
                pickle.dump({ 'best_grids': best_grids,
                              'grids_11': grids_11,
                              'grids_12': grids_12,
                              'current_grids': t_grids
                            }, fout)

        if i_episode % 10 == 0:
            log.info("episode: %s/%s, steps: %s, eps: %.4f, reward: %.4f, connected: %s" %
                    (i_episode, num_episodes, t_step, agent.epsilon, t_reward, env.total_connected))

        if i_episode >= num_episodes-1 or i_episode % 2000 == 0:
            agent.save("./trained_models/connect_dots_dqn_gpu_preferdots.h5")
            with open('./saved_stats/connect_dots_dqn_gpu_stat_preferdots.pkl', 'wb') as fout:
                pickle.dump({'rewards': all_rewards, 'timesteps': all_timesteps,
                             'all_connected': all_connected},
                            fout)


if __name__ == '__main__':
    turtle_env = ConnectDotsEnv(digit=3, save_grid_on_done=False, max_steps=3000, reward_type='prefer_dots')
    turtle_env.seed()
    train_agent(turtle_env, num_episodes=51000, steps_per_episode=3000, batch_size=32)
