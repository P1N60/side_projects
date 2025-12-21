import numpy as np
import pandas as pd

class Party:
    def __init__(self, party: str):
        self.party = party
        self.data = pd.read_csv("voxmeter.csv")
        self.party_data = self.data[self.data["party_letter"] == party]
        self.current_votes = self.data.get("votes").to_list()[-1]

class Block:
    def __init__(self, parties: str):
        self.parties = parties
        self.data = pd.read_csv("voxmeter.csv")
        self.current_votes = 0.0
    
    def vote_history(self):
        block_vote_history = []
        party_vote_histories = []
        for party in self.parties:
            party_class = Party(party)
            party_vote_histories.append(party_class.party_data.get("votes").to_list())
        for j in range(len(party_class.party_data)):
            sum = 0
            for i in range(len(self.parties)):
                sum += party_vote_histories[i][j]
            block_vote_history.append(round(sum, 2))
        block_vote_history = pd.DataFrame([{"votes": poll} for poll in block_vote_history])
        return block_vote_history

