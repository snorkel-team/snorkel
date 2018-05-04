import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .rnn_base import RNNBase

class LSTM(RNNBase):
    
    def build_model(self, hidden_dim=50, num_layers=1, dropout=0.25, **kwargs):
        bidirectional = False
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True
                            )
        self.output_layer = nn.Linear(hidden_dim, self.cardinality-1)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        
    def forward(self, X, hidden_state):
        # TODO: Make this better
        seq_lengths = torch.zeros((X.size(0)), dtype=torch.long)
        for i in range(X.size(0)):
            for j in range(X.size(1)):
                if X[i, j] == 1:
                    seq_lengths[i] = j
                    break
                seq_lengths[i] = X.size(1)
            #seq_lengths[i] = (X[i, :] == 1).nonzero()[0] if  (X[i, :] == 1).nonzero().size(0) != 0 else X.size(1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx, :]

        encoded_X = self.embedding(X)
        encoded_X = pack_padded_sequence(encoded_X, seq_lengths, batch_first=True)
        output, _ = self.lstm(encoded_X, hidden_state)
        output, _ = pad_packed_sequence(output, batch_first=True)

        outs = []
        for i in range(X.size(0)):
            outs.append(output[i, seq_lengths[i] - 1,:])
        output = torch.stack(outs, dim=0)
        return self.output_layer(self.dropout_layer(output))
    
    def initalize_hidden_state(self, batch_size):
        return (
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim),
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        )
            