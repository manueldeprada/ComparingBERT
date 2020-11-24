#!/usr/bin/env python3
import ast
import hashlib
import os
import argparse, logging
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import spearmanr
import file_utils
from algoritmo_bert import algoritmo_bert
from algoritmo_bert_sentence import algoritmo_bert_sentence


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    logging.info("Reading file: %s." % args.evaluationfile)
    logging.getLogger('matplotlib.font_manager').disabled = True  # bug tonto de matplotlib
    tuplas, puntos_test = file_utils.readFile(args.evaluationfile)
    cache_testfile_id = hashlib.md5(open(args.evaluationfile, 'rb').read()).hexdigest() + args.evaluationfile

    algoritmos = [algoritmo_bert_sentence(tuplas, premodel='bert-large-nli-mean-tokens'),
                  algoritmo_bert_sentence(tuplas, premodel='bert-base-nli-mean-tokens'),
                  algoritmo_bert_sentence(tuplas, premodel='bert-base-nli-stsb-mean-tokens'),
                  algoritmo_bert_sentence(tuplas, premodel='roberta-base-nli-mean-tokens'),
                  algoritmo_bert_sentence(tuplas, premodel='distilbert-base-nli-mean-tokens'),
                  algoritmo_bert(tuplas, POS=2, bert_version='bert-large-uncased'),
                  algoritmo_bert(tuplas, POS=2, bert_version='bert-base-uncased'),
                  algoritmo_bert(tuplas, POS=2, bert_version='roberta-base'),
                  algoritmo_bert(tuplas, POS=2, bert_version='distilbert-base-uncased'),
                  algoritmo_bert(tuplas, POS=1, bert_version='bert-large-uncased'),
                  algoritmo_bert(tuplas, POS=1, bert_version='bert-base-uncased'),
                  algoritmo_bert(tuplas, POS=1, bert_version='roberta-base'),
                  algoritmo_bert(tuplas, POS=1, bert_version='distilbert-base-uncased'),
                  algoritmo_bert(tuplas, POS=3, bert_version='bert-large-uncased'),
                  algoritmo_bert(tuplas, POS=3, bert_version='bert-base-uncased'),
                  algoritmo_bert(tuplas, POS=3, bert_version='roberta-base'),
                  algoritmo_bert(tuplas, POS=3, bert_version='distilbert-base-uncased'),

                  ]

    for alg in algoritmos:
        print("\nTesting: " + str(alg))
        cache = 'cache/' + str(alg) + "-" + cache_testfile_id
        if os.path.isfile(cache):  # if we have the result for this test file and algorithm cached, we use it
            with open(cache, "r") as outfile:
                resultado_algoritmo = ast.literal_eval(outfile.read())
        else:
            resultado_algoritmo = alg.process_tuples()
            with open(cache, "w") as outfile:
                outfile.write(str(resultado_algoritmo))

        plt.ylim(0.1, 1.1)
        plt.scatter(puntos_test, resultado_algoritmo)  # y=resultado, x=referencia
        plt.savefig('plots/' + str(alg) + '_' + args.evaluationfile + '.png')
        plt.close()

        rPearson = np.corrcoef(puntos_test, resultado_algoritmo)[0][1]
        rSpearman, p = spearmanr(puntos_test, resultado_algoritmo)
        print("\tpearson=" + str(rPearson) + ",spearman=" + str(rSpearman) + "\n\tcorr_mean=" + str(
            (rPearson + rSpearman) / 2.0))

    # "semval_score=" + str(  2.0 * rPearson * rSpearman / (rPearson + rSpearman)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Flexible NLP sense comparator.",
        epilog="As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
        fromfile_prefix_chars='@')
    # parser.add_argument(
    #     "argument",
    #     help="argumento posicional",
    #     metavar="ARG")
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    parser.add_argument("--evaluationfile", "-f", type=str, required=True)
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)
