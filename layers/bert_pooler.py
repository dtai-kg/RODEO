import torch
import torch.nn as nn

class BertMultiPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_outputs = self.dense(hidden_states)
        pooled_outputs = self.activation(pooled_outputs)
        return pooled_outputs

class BertMultiPairPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_first_cls = hidden_states[:, 0].unsqueeze(1).repeat([1, hidden_states.shape[1], 1])
        pooled_outputs = self.dense(torch.cat([hidden_states_first_cls, hidden_states], 2))
        pooled_outputs = self.activation(pooled_outputs)
        return pooled_outputs