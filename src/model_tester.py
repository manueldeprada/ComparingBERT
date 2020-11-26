from abc import ABC, abstractmethod


# Abstract class: concrete implementations of the models should override the methods.
class ModelTester(ABC):
    def __init__(self, pairs):
        """

        :type pairs: pairs that have the form (a,b).
                      We will return the similarity between a and b.
        """
        self.pairs = pairs

    @abstractmethod
    def process_pairs(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
