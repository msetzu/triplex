from typing import List

import numpy
import torch
import nltk

import pathlib

from sentence_transformers import SentenceTransformer, util

from similarities_modules.infersent.models import InferSent


class Similarity:
    @staticmethod
    def similarity(embeddings: torch.Tensor, metric: str = 'cosine') -> float:
        """
        Compute the similarity between the given `embeddings` according to `distance` metric.
        Args:
            embeddings: Embeddings whose similarity to compare.
            metric: Distance to use, available distances are 'cosine', 'dot' (dot product), and euclidean.

        Returns:
            Upper-triangular similarity matrix of similarities. Lower-triangular entries are
            set to numpy.nan.
        """
        if metric == 'cosine':
            similarity_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
        elif metric == 'dot':
            similarity_scores = numpy.array([numpy.array([torch.dot(emb1, emb2) for emb2 in embeddings])
                                             for emb1 in embeddings])
        elif metric == 'euclidean':
            similarity_scores = numpy.array([torch.sqrt(torch.square(emb - embeddings).sum(axis=1)).numpy()
                                            for emb in embeddings])
            similarity_scores = 1 / similarity_scores

        for i in range(similarity_scores.shape[0]):
            similarity_scores[i:, i] = numpy.nan

        return similarity_scores


class SBERT:
    def __init__(self, model: str = 'stsb-distilbert-base'):
        self.model = SentenceTransformer(model)

    def similarity(self, *texts: str, metric: str = 'cosine') -> numpy.array:
        """
        Compute the similarity between the given `texts`.
        Args:
            metric: Distance to use, available distances are 'cosine', 'dot' (dot product), and euclidean.
            texts: Text whose similarity to compare.

        Returns:
            Upper-triangular similarity matrix of similarities. Lower-triangular entries are
            set to numpy.nan.
        """
        texts_list = list(texts)
        embeddings = self.model.encode(texts_list, convert_to_tensor=True)

        return Similarity.similarity(embeddings, metric=metric)


class Infersent:
    def __init__(self):
        # nltk.download('punkt')
        WD = str(pathlib.Path().absolute()) + '/'
        infersent_path = WD + '../similarities_modules/infersent/'
        embeddings_path = WD + '../similarities_modules/infersent/fastText/crawl-300d-2M.vec'
        MODEL_PATH = infersent_path + 'encoder/infersent2.pkl'
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max',
                        'dpout_model': 0.0, 'version': 2}

        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.set_w2v_path(embeddings_path)

    def similarity(self, *texts: str, metric: str = 'cosine') -> numpy.array:
        """
        Compute the similarity between the given `texts`.
        Args:
            texts: Text whose similarity to compare.
            metric: Distance to use, available distances are 'cosine', 'dot' (dot product), and euclidean.

        Returns:
            Upper-triangular similarity matrix of similarities. Lower-triangular entries are
            set to numpy.nan.
        """
        texts_list = list(texts)
        self.model.build_vocab(texts_list, tokenize=True)
        embeddings = self.model.encode(texts_list, tokenize=True)
        embeddings = torch.Tensor(embeddings)

        return Similarity.similarity(embeddings, metric=metric)
