from abc import ABC, abstractmethod

#Abstract class: concrete implementations of algorithms should override the methods.
class algoritmo(ABC):
    def __init__(self, tuples):
        """

        :type tuples: tuples that have the form (a,b,c...).
                      We will return the similarity between a,b,c...
        """
        self.tuples=tuples


    @abstractmethod
    def process_tuples(self):
        pass
