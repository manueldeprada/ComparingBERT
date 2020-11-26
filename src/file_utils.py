import re


def read_file(dataset_file):
    pairs = []
    dataset_similarities = []
    with open(dataset_file, 'r') as f:
        for line in f:
            regexp = re.compile(r'(.+(?:@.+)+) (.+(?:@.+)) (\d*\.?\d*)\n')
            if re.match(regexp, line):
                groups = re.search(regexp, line).groups()
                pairs.append((groups[0], groups[1]))
                dataset_similarities.append(float(groups[2]))
    return pairs, dataset_similarities
