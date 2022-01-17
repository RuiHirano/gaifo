
class ReplayMemory:
    def __init__(
        self,
        capacity: int = 0,
        batch_size: int = 0
    ):
        self._capacity = capacity
        self._batch_size = batch_size
        self._transactions = []

    def add_transitions(self, transitions: List[Transition]):
        self._transactions.extend(transitions)

    def sample_transitions(self):
        trs = random.sample(self._transactions, self._batch_size)
        return trs

    def __len__(self):
        return len(self._transactions)