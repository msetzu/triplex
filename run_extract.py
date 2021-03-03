import fire as fire
from transformers import AutoModelForSequenceClassification, AutoModel

import numpy
import pandas as pd

import logzero

import json
import time

from exceptions import ModelInferenceError

# base folder for data
from triplex import TripleX

BASE_FOLDER = './data/'


def to_standard_labels(labels, dataset):
    """Standardize different labels to NLI format" -1 for contradiction, 0 for neutrality and +1 for entailment."""
    if dataset == 'mnli':
        return numpy.vectorize(lambda x: -1 if x == 'contradiction' else 0 if x == 'neutral' else 1)(labels)
    elif dataset in {'axb', 'cb', 'axg', 'qnli'}:
        return numpy.vectorize(lambda x: 1 if x == 0 else 1)(labels)
    elif dataset == 'rte':
        return labels

    return None


def extract(dataset: str, model: str, depth: int = 1, width: int = -1, max_perturbations: int = 5,
            max_perturbations_per_token: int = 5, output: str = '', loglevel: str = 'info'):
    """
    Extract explanations for model `model` on data stored in `dataset`.
    Args:
        dataset: Path to the jsonl data
        model: Huggingface model. See 'https://huggingface.co/models' for a complete list
        depth: Depth of hypernym perturbation. Each consecutive hypernym increases depth by 1. Defaults to 1
        width: Width of perturbation: how many tokens to perturb? Defaults to -1 (no limit)
        max_perturbations: Maximum number of perturbations to generate.
        max_perturbations_per_token: Maximum number of perturbations per token.
        output: Output file where to dump the output.
        loglevel: Logging level, any of 'debug', 'info', 'error'.

    Returns:

    """
    if model in {'textattack/roberta-base-RTE', 'roberta-large-mnli'}:
        transformer = AutoModelForSequenceClassification.from_pretrained(model, output_attentions=True)
    else:
        transformer = AutoModel.from_pretrained(model, output_attentions=True)

    if output == '':
        output_file = dataset + '_' + time.asctime()
    else:
        output_file = output

    # logs
    if loglevel == 'info':
        logzero.loglevel(logzero.logging.INFO)
    elif loglevel == 'debug':
        logzero.loglevel(logzero.logging.DEBUG)
    elif loglevel == 'error':
        logzero.loglevel(logzero.logging.ERROR)

    # load data
    data = pd.read_json(dataset, lines=True)
    data = data.drop('idx', axis='columns')
    data['label'] = to_standard_labels(data['label'].values, dataset)
    data = data[['premise', 'hypothesis', 'label']]

    # logs
    logzero.logger.info('Dataset: ' + dataset.lower())
    logzero.logger.info('Model: ' + model)
    logzero.logger.info('Output: ' + output_file)

    i = 0
    for idx, row in data.iterrows():
        print(i)
        i += 1
        premise, hypothesis, label = row.premise, row.hypothesis, row.label
        try:
            # explainer
            gen = TripleX(transformer)
            explanations = gen.extract(premise, hypothesis, depth=depth, width=width,
                                       max_perturbations=max_perturbations,
                                       max_perturbations_per_token=max_perturbations_per_token)
            explanation_json = list()
            for i, (perturbed_dfa, hypothesis, perturbation_distance, perturbation_pairs) in enumerate(explanations):
                explanation_json.append([perturbed_dfa.to_json(), hypothesis, perturbation_distance,
                                         list(map(list, perturbation_pairs))])
            dump = [idx, premise, hypothesis, explanation_json]
            logzero.logger.debug('Dumping to ' + output_file + '.jsonl')
            with open(output_file + '.jsonl', 'a+') as log:
                json.dump(dump, log, indent=0)
        except ModelInferenceError:
            logzero.logger.info('Model could not infer.')


if __name__ == '__main__':
    fire.Fire(extract)
