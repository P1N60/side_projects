class party:
    def __init__(self, data, party):
        self.party = party
        self.party_data = data[data["party_letter"] == party]
        self.votes = self.party_data.get("votes")[0]