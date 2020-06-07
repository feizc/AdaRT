import torch.nn as nn
import torch.nn.functional as F
import torch


class FeedForward(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_sizes=(512,),
                 activation="Tanh", bias=True, dropout=0.1):
        super(FeedForward, self).__init__()
        self.activation = getattr(nn, activation)()

        n_inputs = [input_dim] + list(hidden_sizes)
        n_outputs = list(hidden_sizes) + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias)
                                      for n_in, n_out in zip(n_inputs, n_outputs)])
        self.num_layer = len(self.linears)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_):
        x = input_
        i = 0
        for linear in self.linears:
            x = linear(x)
            if i < self.num_layer - 1:
                x = self.dropout_layer(x)
            x = self.activation(x)
            i +=1
        return x


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.policy = FeedForward(state_dim, action_dim, hidden_sizes=(128,64))

    def forward(self, state):
        action_score = self.policy(state)
        action_prob = F.softmax(action_score, dim=-1)
        return action_prob


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.critic = FeedForward(state_dim+action_dim, 1, hidden_sizes=(128, 64))

    def forward(self, state_actions):
        val = self.critic(state_actions)
        return val


