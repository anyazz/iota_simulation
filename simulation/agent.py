class Agent:
    def __init__(self, counter):
        self.id = counter
        self.visible_transactions = []
        self.tip_selection_algo = None
        #For analysis
        self.agent_average_confirmation_confidence = 0
        self.tips = []
        self.record_tips = []

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)
