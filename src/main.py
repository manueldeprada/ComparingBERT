#!/usr/bin/env python3
import argparse
import ast
import hashlib
import logging
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import file_utils
from bert_sentence_tester import BertSentenceTester
from bert_tester import BertTester


def main(args):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True

    for file in args.dataset_file:
        logging.info("Reading file: %s." % file)
        pairs, dataset_similarities = file_utils.read_file(file)
        cache_file_id = hashlib.md5(open(file, 'rb').read()).hexdigest() + file

        models = [
            BertSentenceTester(pairs, version='bert-large-nli-mean-tokens'),
            BertSentenceTester(pairs, version='bert-base-nli-mean-tokens'),
            BertSentenceTester(pairs, version='roberta-base-nli-mean-tokens'),
            BertSentenceTester(pairs, version='distilbert-base-nli-mean-tokens'),
            BertTester(pairs, similarity_type='head', version='bert-large-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='head', version='bert-base-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='head', version='roberta-base', combine_method='sum'),
            BertTester(pairs, similarity_type='head', version='distilbert-base-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-subj', version='bert-large-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-subj', version='bert-base-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-subj', version='roberta-base', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-subj', version='distilbert-base-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-obj', version='bert-large-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-obj', version='bert-base-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-obj', version='roberta-base', combine_method='sum'),
            BertTester(pairs, similarity_type='dep-obj', version='distilbert-base-uncased', combine_method='sum'),
            BertTester(pairs, similarity_type='head', version='bert-large-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='head', version='bert-base-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='head', version='roberta-base', combine_method='concat'),
            BertTester(pairs, similarity_type='head', version='distilbert-base-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-subj', version='bert-large-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-subj', version='bert-base-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-subj', version='roberta-base', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-subj', version='distilbert-base-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-obj', version='bert-large-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-obj', version='bert-base-uncased', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-obj', version='roberta-base', combine_method='concat'),
            BertTester(pairs, similarity_type='dep-obj', version='distilbert-base-uncased', combine_method='concat'),
        ]

        for model in models:
            # Do not launch dep-obj focused models in NV datasets.
            if len(pairs[0][0].split('@')) < 3 and \
                    hasattr(model, 'similarity_type') and model.similarity_type == 'dep-obj':
                continue
            print("Model: " + str(model) + ", dataset: " + file)
            cache = 'cache/' + str(model) + "-" + cache_file_id
            if os.path.isfile(cache):  # We look for the file of cached similarities.
                with open(cache, "r") as outfile:
                    model_similarities = ast.literal_eval(outfile.read())
            else:
                model_similarities = model.process_pairs()
                with open(cache, "w") as outfile:  # Write the cache file for the future
                    outfile.write(str(model_similarities))

            pearson_corr, _ = pearsonr(dataset_similarities, model_similarities)
            spearman_corr, _ = spearmanr(dataset_similarities, model_similarities)
            mean = "{:.2f}".format(100.0 * 2.0 * pearson_corr * spearman_corr / (pearson_corr + spearman_corr))
            if pearson_corr * spearman_corr < 0:  # If not both positive or negative, standard mean instead of harmonic
                mean = "{:.2f}".format(100.0 * (pearson_corr + spearman_corr) / 2.0) + '*'
            print("\r\tPearson=" + "{:.2f}".format(100.0 * pearson_corr) + ", Spearman=" + "{:.2f}".format(
                100.0 * spearman_corr) + ", Mean=" + mean)

            # plt.ylim(0.1, 1.1)  # Scale for the plots
            # plt.scatter(dataset_similarities, model_similarities)
            # plt.savefig('plots/' + str(model) + '-' + file + '.png')
            # plt.close()

            dataset_standarized = np.array(dataset_similarities)
            dataset_standarized = (dataset_standarized - np.mean(dataset_standarized)) / np.std(dataset_standarized)
            model_standarized = np.array(model_similarities)
            model_standarized = (model_standarized - np.mean(model_standarized)) / np.std(model_standarized)
            plt.xlim(-1.5, 2.15)  # Scale for the plots
            plt.ylim(-2.5, 2.5)  # Scale for the plots
            plt.scatter(dataset_standarized, model_standarized)
            plt.savefig('plots_standarized/' + str(model) + '-' + file + '.png')
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A flexible sentence similarity models comparator with a cache for already tested models.",
        epilog="See README on github.com/....")
    parser.add_argument("dataset_file", type=str, nargs='+',
                        help="File path of one or more dataset files to execute the models."
                             "Dataset file should contain pairs of NV or NVN expressions "
                             "following Mitchell and Lapata (2008) format.")
    main(parser.parse_args())
