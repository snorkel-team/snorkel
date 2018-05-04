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
                            dropout=dropout if num_layers > 1 else 0, batch_first=True
                            )
        self.output_layer = nn.Linear(hidden_dim, self.cardinality-1)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        
    def forward(self, X, hidden_state):
        X_order = sorted(
            [(idx, int(max(candidate.nonzero())[0].data)) 
            for idx, candidate in enumerate(X)
            ],
            reverse=True, 
            key=lambda x: x[1]
        )

        X_sorted_order, X_lens = zip(*X_order)
        X_sorted_order, X_lens = list(X_sorted_order), list(X_lens)
        encoded_X = self.embedding(X)
        encoded_X = pack_padded_sequence(encoded_X[X_sorted_order,:,:], X_lens, batch_first = True)
        output, _ =  self.lstm(encoded_X, hidden_state)
        output, _ = pad_packed_sequence(output, batch_first = True)
        temp_output = list(
            map(lambda x: x[1],
                    sorted(
                    [(idx, data[lens-1, :]) for idx, lens, data in zip(X_sorted_order, X_lens, output)],
                    key=lambda x: x[0]
                    )
               )
        )
        output = torch.stack(temp_output)
        output_layer = nn.Linear(output.size(1), self.cardinality-1)
        return output_layer(self.dropout_layer(output))
    
    def initalize_hidden_state(self, batch_size):
        return (
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim),
            torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        )
            