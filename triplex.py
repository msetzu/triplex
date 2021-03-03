import string
from abc import abstractmethod
from typing import Union, List, Tuple, Callable

import logzero
import numpy
import scipy
import torch
from nltk.corpus import stopwords
from transformers import AutoModelForSequenceClassification, RobertaTokenizer

from dfas import DFAH
from exceptions import ModelInferenceError
from parsers.ie import OpenIEParser
from perturbations import HypernymPerturbator


class Observer:
    def __init__(self, model: AutoModelForSequenceClassification):
        self.model = model

    def attention(self, premise: str, hypothesis: Union[str, None] = None, use_stop_words: bool = False) -> Tuple[torch.Tensor, int]:
        """Compute attention scores of `model` on `s0, s1`."""
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        x = tokenizer.encode_plus(premise, hypothesis, add_special_tokens=True, return_tensors='pt')
        logzero.logger.debug('Model inference...')
        try:
            label, attentions = self.model(x['input_ids'])[-2:]
        except IndexError:
            raise ModelInferenceError()
        logzero.logger.debug('Model inference done.')
        # Remove first dimension for single prediction
        attentions = torch.cat(attentions, dim=0)
        label = label.argmax().item()

        # Do not pay attention (VERY FUN PUN) to some tokens
        ignore_set = {'</s>', '<s>', '[CLS]', '[SEP]'}
        if not use_stop_words:
            ignore_set = ignore_set.union(set(stopwords.words('english')))
            ignore_set = ignore_set.union(set(string.punctuation))

            # Actual tokens
        if hypothesis is not None:
            tokens = tokenizer.encode_plus(premise, hypothesis, add_special_tokens=True, return_tensors='pt')['input_ids'][0].numpy()
        else:
            tokens = tokenizer.encode_plus(premise, add_special_tokens=True, return_tensors='pt')['input_ids'][0].numpy()
            tokens = [tokenizer.decode(int(i), skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False).replace(' ', '')
                      for i in tokens]

        ignore_idx = {i for i, t in enumerate(tokens) if t in ignore_set}
        # Remove words in the ignore list
        for attention_layer in attentions:
            for head in attention_layer:
                for i in ignore_idx:
                    head[i, :] = numpy.nan
                    head[:, i] = numpy.nan
                    # Remove self-attention
                    for j in range(head.shape[0]):
                        head[j, j] = numpy.nan

        return attentions, label


class TriplesGenerator:
    @abstractmethod
    def extract(self, premise: str, hypothesis: str, **kwargs):
        pass


class TripleX(TriplesGenerator):
    """Generates a set of candidates explanation for a given model and input.
    Use method `generate(inp: str) -> Set[Tuple(DFA, float)]` to get a set of candidate explanations
    in DFA form, each associated with an explanation score: the higher the score, the better the explanation.
    """
    def __init__(self, model):
        self.model = model
        self.observer = Observer(model)
        self.perturbator = HypernymPerturbator()
        self.parser = OpenIEParser()

    def extract(self, premise: str, hypothesis: str, **kwargs) -> Tuple[List[DFAH], List[DFAH]]:
        """
        Generate a set of DFAs to explain the given input.
        Args:
            premise: The premise
            hypothesis: The hypothesis
            kwargs:
                width: Maximum number of tokens to perturb. Defaults to -1 (no limit)
                depth: Maximum hypernyms substitutions for any token. Defaults to -1 (no limit)
                max_distance: Maximum hypernym distance. Every hypernym substitution increases distance by 1.
                norm (Union, optional): Norm to use. str ('fro', 'inf', '-inf') or int for classic norms, or a
                            distance function(float, float) -> float instead. Defaults to 'fro' (Frobenius).

        Returns:
            The DFAs
        """
        width = kwargs.get('width', -1)
        depth = kwargs.get('depth', -1)
        max_distance = kwargs.get('max_distance', -1)
        norm = kwargs.get('norm', 'fro')
        max_perturbations = kwargs.get('max_perturbations', 1000)
        max_perturbations_per_token = kwargs.get('max_perturbations_per_token', 3)

        logzero.logger.debug('Extracting triples...')
        dfa = self.parser.parse(premise)
        logzero.logger.debug('Extracting reference attention...')
        try:
            attention_graph, label = self.observer.attention(dfa.to_text(), hypothesis)
        except ModelInferenceError:
            raise ModelInferenceError
        logzero.logger.debug('Attention gathered...')
        attention_graph = attention_graph[-1].detach().numpy()
        # reference attention matrix to compute distances
        attention_matrix = numpy.nanmean(attention_graph, axis=0)

        # compute perturbations
        logzero.logger.debug('Perturbing...')
        perturbed_dfas = self.perturbator.perturb(dfa, max_width=width, max_depth=depth,
                                                           max_distance=max_distance,
                                                           max_perturbations_per_token=max_perturbations_per_token,
                                                           max_perturbations=max_perturbations)
        logzero.logger.debug('Perturbed.')

        perturbations = list()
        for i, perturbed_premise in enumerate(perturbed_dfas):
            logzero.logger.debug('Extracting attentions, {0}/{1}...'.format(i, len(perturbed_dfas)))
            try:
                perturbed_attention_graph, perturbed_label = self.observer.attention(perturbed_premise.to_text(),
                                                                                     hypothesis)
            except IndexError:
                continue
            perturbed_attention_graph = perturbed_attention_graph[-1].detach().numpy()
            perturbed_attention_matrix = numpy.nanmean(perturbed_attention_graph, axis=0)

            # align matrices in case the hypernym has a larger tokenization
            if perturbed_attention_matrix.shape[0] != attention_matrix.shape[0]:
                perturbed_attention_matrix = self.perturbator.align_attention_matrices(attention_matrix,
                                                                                       perturbed_attention_matrix,
                                                                                       perturbed_premise.to_text(),
                                                                                       hypothesis)

            perturbation_distance = self._attention_distance(attention_matrix, perturbed_attention_matrix, norm=norm)
            perturbations.append((perturbed_premise, perturbed_label, float(perturbation_distance)))

        logzero.logger.debug('Extracted, ranking perturbations...')
        concordant_dfas = [(p, distance, perturbation_pairs)
                           for (p, p_label, distance, perturbation_pairs) in perturbations if p_label == label]
        discordant_dfas = [(p, distance, perturbation_pairs)
                           for (p, p_label, distance, perturbation_pairs) in perturbations if p_label != label]
        concordant_dfas = sorted(concordant_dfas, key=lambda x: x[2])
        concordant_dfas = [dfa for dfa, _, _ in concordant_dfas]
        discordant_dfas = sorted(discordant_dfas, key=lambda x: x[2])
        discordant_dfas = [dfa for dfa, _, _ in discordant_dfas]
        logzero.logger.debug('Ranked.')

        return concordant_dfas, discordant_dfas

    @staticmethod
    def _attention_distance(A: numpy.array, B: numpy.array,
                            norm: Union[Callable[[float, float], float], str, int] = 'fro') -> float:
        """Return the distance between the two attention matrices `A` and `B`.

        Args:
            A (numpy.array): Attention matrix.
            B (numpy.array): Attention matrix.
            norm (Union, optional): Norm to use. str ('fro', 'inf', '-inf') or int for classic norms, or a
                                    distance function(float, float) -> float instead. Defaults to 'fro' (Frobenius).

        Returns:
            float: Distance between the two attention matrices.
        """
        for i in range(A.shape[0]):
            A[i, i] = 1.
            B[i, i] = 1.
        if isinstance(norm, str) and norm in {'fro', 'inf', '-inf'} or isinstance(norm, int):
            D = A - B
            distance = scipy.linalg.norm(D, ord=norm)
        elif isinstance(norm, Callable):
            # Hadamard-like distance, sum of pairwise distances
            distance = sum(norm(a, b) for a, b in zip(A.flatten(), B.flatten()))
        else:
            raise ValueError('norm of type ' + str(type(norm)) + ', expected str, int or'
                                                                 'function(float, float) -> float')

        return distance
