import fire as fire
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModel, logging

import numpy
import pandas as pd

import logzero

import json
import time
import os
import sys
import pathlib

# add this module to path
BASE_FOLDER = str(pathlib.Path().absolute()) + '/../'
sys.path.append(BASE_FOLDER)

from exceptions import ModelInferenceError
from triplex import TripleX

# shut up transformers!
logging.set_verbosity_error()


def to_standard_labels(labels, dataset):
    """Standardize different labels to NLI format" -1 for contradiction, 0 for neutrality and +1 for entailment."""
    if dataset == 'mnli':
        return numpy.vectorize(lambda x: -1 if x == 'contradiction' else 0 if x == 'neutral' else 1)(labels)
    elif dataset in {'axb', 'cb', 'axg', 'qnli'}:
        return numpy.vectorize(lambda x: 1 if x == 0 else 1)(labels)
    elif dataset == 'rte':
        return labels

    return None


def extract(dataset: str, model: str, depth: int = 1, width: int = -1, max_perturbations: int = -1,
            max_perturbations_per_token: int = 5, parser: str = 'openie', layer: int = -1, head: int = -1,
            output: str = '', loglevel: str = 'info', port: int = 9000):
    """
    Extract explanations for model `model` on data stored in `dataset`.
    Args:
        dataset: Path to the jsonl data
        model: Huggingface model. See 'https://huggingface.co/models' for a complete list
        depth: Depth of hypernym perturbation. Each consecutive hypernym increases depth by 1. Defaults to 1
        width: Width of perturbation: how many tokens to perturb? Defaults to -1 (no limit)
        max_perturbations: Maximum number of perturbations to generate.
        max_perturbations_per_token: Maximum number of perturbations per token.
        parser: Parser to use. Choose one of 'openie', 'ollie' or 'clausie'.
        layer: Layer to consider for alignment scoring. Defaults to -1 (last layer)
        head: Head to consider for alignment scoring. Defaults to -1 (all heads in layer)
        output: Output file where to dump the output.
        loglevel: Logging level, any of 'debug', 'info', 'error'.
        port: Port for the triplex server. Defaults to 9000.

    Returns:

    """
    if model in {'textattack/roberta-base-RTE', 'roberta-large-mnli'}:
        transformer = AutoModelForSequenceClassification.from_pretrained(model, output_attentions=True)
    else:
        transformer = AutoModel.from_pretrained(model, output_attentions=True)
    transformer.training = False

    if output == '':
        output_file = str(pathlib.Path(__file__).parent.absolute()) + '/' + dataset + '_' + time.asctime()
    else:
        output_file = str(pathlib.Path(__file__).parent.absolute()) + '/' + output

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
    logzero.logger.info('Parser: ' + parser)
    logzero.logger.info('Output: ' + output_file)

    # check if the output file has already been partially created
    if os.path.isfile(output_file + '.jsonl'):
        with open(output_file + '.jsonl', 'r') as log:
            skip_lines = len(log.readlines())
        data = data.iloc[skip_lines:]
        i = skip_lines
    else:
        i = 0
    explainer = TripleX(transformer, parser=parser, port=port)
    logzero.logger.info('Extracting explanations...')
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        i += 1
        premise, hypothesis, label = row.premise, row.hypothesis, row.label
        try:
            # explainer
            explanations, counterfactual_explanations = explainer.extract(premise, hypothesis, depth=depth, width=width,
                                                                          max_perturbations=max_perturbations,
                                                                          max_perturbations_per_token=max_perturbations_per_token,
                                                                          layer=layer, head=head)
            explanation_json = {
                'dataset': dataset.lower(),
                'model': model,
                'layer': layer,
                'head': head,
                'parser': parser,
                'depth': depth,
                'width': width,
                'max_perturbations': max_perturbations,
                'max_perturbations_per_token': max_perturbations_per_token,
                'dataset_idx': i,
                'premise': premise,
                'hypothesis': hypothesis,
                'explanations': [e.to_json() for e in explanations],
                'counterfactual_explanations': [e.to_json() for e in counterfactual_explanations]
            }
            logzero.logger.debug('Dumping to ' + output_file + '.jsonl')
            with open(output_file + '.jsonl', 'a+') as log:
                json.dump(explanation_json, log)
                log.write('\n')
        except ModelInferenceError:
            logzero.logger.info('Model could not infer.')

    logzero.logger.info('Done.')


if __name__ == '__main__':
    fire.Fire(extract)
