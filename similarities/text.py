import numpy
import torch

from sentence_transformers import SentenceTransformer, util


class TransformerSimilarity:
    def __init__(self, model: str = 'stsb-distilbert-base'):
        self.model = SentenceTransformer(model)

    def similarity(self, distance='cosine', *texts: str) -> numpy.array:
        """
        Compute the similarity between the given `texts`.
        Args:
            distance: Distance to use, available distances are 'cosine', 'dot' (dot product), and euclidean.
            texts: Text whose similarity to compare.

        Returns:
            Upper-triangular similarity matrix of similarities. Lower-triangular entries are
            set to numpy.nan.
        """
        texts_list = list(texts)
        n = len(texts_list)
        embeddings = self.model.encode(texts_list, convert_to_tensor=True)

        if distance == 'cosine':
            similarity_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()
        elif distance == 'dot':
            similarity_scores = numpy.array([numpy.array([torch.dot(emb1, emb2) for emb2 in embeddings])
                                             for emb1 in embeddings])
        elif distance == 'euclidean':
            similarity_scores = numpy.array([torch.sqrt(torch.square(emb - embeddings).sum(axis=1)).numpy()
                                            for emb in embeddings])
            similarity_scores = 1 / similarity_scores

        for i in range(n):
            similarity_scores[i:, i] = numpy.nan

        return similarity_scores
