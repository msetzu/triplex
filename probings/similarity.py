from typing import Tuple, List

import fire
import numpy
import pandas

import time
import pathlib
import sys

sys.path.append(str(pathlib.Path().absolute()) + '/../')

from similarities.text import SBERT, Infersent


def dataset_similarity(dataset: str, model: str = 'stsb-distilbert-base', metric: str = 'cosine') -> Tuple[List[str], numpy.array]:
    """
    Compute similarities from `dataset` using model `model`.

    Args:
        dataset: Path to the dataset, a JSONL with a
        model: Name of the sentence-transformer model to use.
        metric: Distance to use, available distances are 'cosine', 'dot' (dot product), and euclidean.

    """
    # load data
    data = pandas.read_json(dataset, lines=True)
    data = data.drop('idx', axis='columns')
    data = data['premise'].values.tolist()

    if model == 'infersent':
        similarity_box = Infersent()
    else:
        similarity_box = SBERT(model)
    similarity_matrix = similarity_box.similarity(*data, metric=metric)
    assert len(data) == similarity_matrix.shape[0]

    return data, similarity_matrix


def main(dataset: str, model: str = 'stsb-distilbert-base', output: str = ''):
    """
    Compute similarities from `dataset` using model `model`.
    Args:
        dataset: Path to the dataset, a JSONL with a
        model: Name of the sentence-transformer model to use.
        output: Path to dump the analysis result
    """
    if output == '':
        output_file = str(pathlib.Path(__file__).parent.absolute()) + '/' + dataset + '_' + time.asctime() + '.dat'
    else:
        output_file = str(pathlib.Path(__file__).parent.absolute()) + '/' + output + '.dat'

    _, similarity_matrix = dataset_similarity(dataset, model)

    numpy.matrix.dump(similarity_matrix, output_file)


if __name__ == '__main__':
    fire.Fire(main)
