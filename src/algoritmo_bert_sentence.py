from algoritmo import algoritmo
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class algoritmo_bert_sentence(algoritmo):
    def __init__(self, tuples, premodel):
        """

        :type tuples: tuples that have the form (a,b,c...).
                      We will return the similarity between a,b,c...
        """
        super().__init__(tuples)
        self.tuples = tuples
        self.model = SentenceTransformer(premodel)
        self.name=premodel

    def __str__(self):
        return "sentence_"+self.name

    def process_tuples(self):
        puntos = []
        i = 0
        for tupla in self.tuples:
            puntos.append(self.procesar_tupla(tupla))
            i += 1
            print("\rProgress {:2.1%}".format(i / len(self.tuples)), end='')
        return puntos

    def procesar_tupla(self, tupla):
        sent1 = tupla[0].replace("@", " ")
        sent2 = tupla[1].replace("@", " ")

        # Sentences we want sentence embeddings for
        sentence1_embedding = self.model.encode(sent1,show_progress_bar=False)
        sentence2_embedding = self.model.encode(sent2,show_progress_bar=False)

        # print("Sentence embeddings:")
        # print(sentence1_embedding)

        simil = 1 - cosine(sentence1_embedding, sentence2_embedding)

        return simil
