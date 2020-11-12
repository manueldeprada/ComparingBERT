class Algoritmo:
    def __init__(self, tuples):
        """

        :type tuples: tuples that have the form (a,b,c...).
                      We will return the similarity between a,b,c...
        """
        self.tuples=tuples

    def procesar_tuplas(self, POS):
        raise NotImplementedError("Please Implement this method")

