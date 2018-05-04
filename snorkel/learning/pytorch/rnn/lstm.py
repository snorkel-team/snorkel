import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .rnn_base import RNNBase

class LSTM(RNNBase):
    
    def build_model(self, hidden_dim=50, num_layers=1, bidirectional=True, dropout=0.25, **kwargs):
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True
                            )
        self.output_layer = nn.Linear(hidden_dim, self.cardinality-1)
        
        
    def forward(self, X, hidden_state):
        X, indicies = X.sort(dim=0, descending=True)
        X_lens = torch.stack([max(candidate.nonzero())[0] for candidate in X])
        encoded_X = self.embedding(X)
        encoded_X = pack_padded_sequence(encoded_X, X_lens, batch_first = True)
        output, _ =  self.lstm(encoded_X, hidden_state)
        output, _ = pad_packed_sequence(output, batch_first = True)
        output = torch.stack([data[X_lens[idx]-1, :] for idx, data in enumerate(output)])
        return self.output_layer(output)
    
    def initalize_hidden_state(self, batch_size):
        return (
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim),
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        )
            