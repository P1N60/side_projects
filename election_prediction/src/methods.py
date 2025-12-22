import numpy as np
import pandas as pd
import torch
from torch import nn

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class Party:
    def __init__(self, party: str):
        self.party = party
        self.data = pd.read_csv("../data/voxmeter.csv")
        self.party_data = self.data[self.data["party_letter"] == party]
        self.current_votes = self.data.get("votes").to_list()[-1] # type: ignore

class Block:
    def __init__(self, parties: str):
        self.parties = parties
        self.data = pd.read_csv("../data/voxmeter.csv")
        self.current_votes = 0.0
    
    def vote_history(self):
        block_vote_history = []
        party_vote_histories = []
        for party in self.parties:
            party_class = Party(party)
            party_vote_histories.append(party_class.party_data.get("votes").to_list()) # type: ignore
        for j in range(len(party_class.party_data)): # type: ignore
            sum = 0
            for i in range(len(self.parties)):
                sum += party_vote_histories[i][j]
            block_vote_history.append(round(sum, 2))
        block_vote_history = pd.DataFrame([{"votes": poll} for poll in block_vote_history])
        return block_vote_history

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, hn, cn