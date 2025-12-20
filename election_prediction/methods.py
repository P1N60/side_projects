class party:
    def __init__(self, data, party):
        self.party = party
        self.party_data = data[data["party_letter"] == party]
        self.current_votes = self.party_data.get("votes").to_list()[len(self.party_data)-1]