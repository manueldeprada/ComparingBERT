#!/usr/bin/env python3

import sys, argparse, logging
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import spearmanr
import file_utils
from AlgoritmoBERT import AlgoritmoBERT
from algoritmo import Algoritmo


def main(args, loglevel):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    logging.info("Reading file: %s." % args.evaluationfile)
    tuplas,puntos_test=file_utils.readFile(args.evaluationfile)
    algoritmo=AlgoritmoBERT(tuplas)
    resultado_algoritmo=algoritmo.procesar_tuplas(2)
    puntos_test_normalizados=(puntos_test - np.min(puntos_test)) / (np.max(puntos_test) - np.min(puntos_test))
    res = "\n".join("{} {}".format(x, y) for x, y in zip(puntos_test_normalizados, resultado_algoritmo))
    # isto son os resultados xunto a cada golden, normalizados

    logging.getLogger('matplotlib.font_manager').disabled = True #bug tonto
    plt.scatter(puntos_test,resultado_algoritmo) #y=resultado
    plt.show() #scatter plot

    rPearson = np.corrcoef(puntos_test, resultado_algoritmo)[0][1]
    rSpearman, p = spearmanr(puntos_test, resultado_algoritmo)
    print("\npearson="+str(rPearson)+",spearman="+str(rSpearman)+"semval_score="+str( 2.0 * rPearson * rSpearman / (rPearson + rSpearman)))


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