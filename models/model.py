import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c.distributions import get_distribution
from a2c.utils import orthogonal


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        All classes that inheret from Policy are expected to have
        a feature exctractor for actor and critic (see examples below)
        and modules called linear_critic and dist. Where linear_critic
        takes critic features and maps them to value and dist
        represents a distribution of actions.        
        """

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        action = self.dist.sample(hidden_actor, deterministic=deterministic)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor, action)
        value = self.critic_linear(hidden_critic)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        hidden_critic, _, states = self(inputs, states, masks)
        value = self.critic_linear(hidden_critic)
        return value

    def evaluate_actions(self, inputs, states, masks, actions):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor, actions)
        value = self.critic_linear(hidden_critic)

        return value, action_log_probs, dist_entropy, states


class CNNPolicy(Policy):
    def __init__(self, num_inputs, action_space, use_gru):
        super(CNNPolicy, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(num_inputs, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()

        self.linear1 = nn.Linear(320, 512)

        if use_gru:
            self.gru = nn.GRUCell(512, 512)

        self.critic_linear = nn.Linear(512, 1)

        self.dist = get_distribution(512, action_space)

        self.train()
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        #self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = F.max_pool2d(self.conv1(inputs),2)
        x = F.relu(x)

        x = F.max_pool2d(self.conv2_drop(self.conv2(x)),2)
        x = F.relu(x)

        #x = self.conv3(x)
        #x = F.relu(x)

        x = x.view(-1, 320)
        x = self.linear1(x)
        x = F.relu(x)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
        return x, x, states
