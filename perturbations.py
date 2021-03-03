"""
Perturbations objects allowing to perturb textual input.
"""
import random

import spacy as spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from transformers import RobertaTokenizer

from nltk.tokenize import word_tokenize

import numpy

import itertools
from copy import copy

from typing import Tuple, List, Dict

from dfas import DFA, DFAH


class HypernymPerturbator:
    """
    Perturbs input by replacing its tokens with their hypernyms, i.e. words with more general meaning.
    For instance, 'cat' with 'feline' or 'living being'.
    """
    nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def wordnet(text: str) -> spacy.tokens.doc.Doc:
        """Tag `text` with Wordnet."""
        return WordnetAnnotator(HypernymPerturbator.nlp.lang)(HypernymPerturbator.nlp(text))

    @staticmethod
    def nltk_hypernyms(text: str) -> Dict[str, set[str]]:
        """
        Compute the NLTK hypernyms of the given `text`. Returns a list of pairs (token, hypernyms).
        Args:
            text: The text to perturb.

        Returns:
            Tokens hypernyms and indexes.
        """
        words = HypernymPerturbator.wordnet(text)
        token_hypernyms_dict = dict()
        for w in words:
            if w.pos_ in {'PROPN', 'PUNCT', 'ADP', 'SYM', 'NUM'}:
                continue
            hypernyms = set()
            synsets = w._.wordnet.synsets()
            if len(synsets) > 0:
                for synset in synsets:
                    for hyper in synset.hypernyms():
                        hypernyms.add(hyper.lemmas()[0].name())
                token_hypernyms_dict[w.text] = hypernyms

        return token_hypernyms_dict

    @staticmethod
    def perturbation_hamming(base: Tuple[str], hypernym: Tuple[str]) -> float:
        return sum(a != b for a, b in zip(base, hypernym))

    @staticmethod
    def perturbation_distance(base: Tuple[str], hypernym: Tuple[str], hypernym_distances: Dict[str, Dict[str, int]]) -> float:
        return sum(hypernym_distances[a][b] for a, b in zip(base, hypernym))

    # noinspection PyTypeChecker
    def perturb(self, dfa: DFA, max_width: int = -1, max_depth: int = -1, max_perturbations_per_token: int = -1,
                max_distance: int = -1, max_perturbations=-1) -> List[DFAH]:
        """
        Perturb the given `text` by replacing tokens with their respective hypernyms.
        Args:
            dfa: The parsed DFA generated by a `parsers.Parser`.
            max_width: Maximum number of tokens to perturb. Defaults to -1 (no limit)
            max_depth: Maximum hypernyms substitutions for any token. Defaults to -1 (no limit)
            max_perturbations_per_token: Maximum hypernym distance per single token. Defaults to -1 (no limit)
            max_distance: Maximum hypernym distance. Every hypernym substitution increases distance by 1.
            max_perturbations: Maximum number of perturbations. Defaults to -1 (no limit)

        Returns:
            Set of perturbations: each element in the list is a perturbation and a dictionary token => perturbed token
            mapping each perturbed token to its perturbation
        """
        depth = max_depth if max_depth > 0 else 1000
        perturbations_per_token = max_perturbations_per_token if max_perturbations_per_token > 0 else numpy.inf
        distance = max_distance if max_distance > 0 else numpy.inf

        text = dfa.to_text()
        text_doc = HypernymPerturbator.nlp(text)
        text_tokens = tuple(token.text for token in text_doc)
        tokens_hypernyms = self.nltk_hypernyms(text)

        # depth dictionary: base tokens at depth 0, every hypernym has the depth
        # of its parent + 1
        hypernym_depth = dict()
        for token, token_hypernyms in tokens_hypernyms.items():
            hypernym_depth[token] = {token: 0}
            for token_hypernym in token_hypernyms:
                hypernym_depth[token][token_hypernym] = 1

        # grow and filter by depth
        token_perturbations = dict()
        for token, token_boundary in tokens_hypernyms.items():
            boundary_set = set(tokens_hypernyms[token])
            token_perturbations[token] = {token} | boundary_set
            next_boundary_set = set()

            # offset by 2 to align with depth dictionary
            for d in range(2, max_depth + 2):
                for hypernym in boundary_set:
                    try:
                        next_boundary_hypernyms_dic = self.nltk_hypernyms(hypernym)
                        for next_boundary_hypernyms in next_boundary_hypernyms_dic.values():
                            for next_boundary_hypernym in next_boundary_hypernyms:
                                hypernym_depth[token][next_boundary_hypernym] = depth
                                token_perturbations[token].add(next_boundary_hypernym)
                                next_boundary_set.add(next_boundary_hypernym)
                    except IndexError:
                        break
                # new boundaries replace old ones
                boundary_set = copy(next_boundary_set)
                next_boundary_set = set()
        base_perturbation_tokens = tuple(token_perturbations.keys())
        flat_perturbations = tuple(token_perturbations[token] for token in base_perturbation_tokens)

        if not numpy.isinf(perturbations_per_token):
            flat_perturbations = tuple(random.sample(perturbations, min(perturbations_per_token, len(perturbations)))
                                       for perturbations in flat_perturbations)

        # filter by maximum number of perturbations
        if max_perturbations != -1:
            flat_perturbations = itertools.islice(itertools.product(*flat_perturbations), max_perturbations)
        else:
            flat_perturbations = itertools.product(*flat_perturbations)

        # filter by width
        if max_width != -1:
            flat_perturbations = filter(lambda p: self.perturbation_hamming(text_tokens, p) <= max_width,
                                        flat_perturbations)
        # filter by distance
        if max_distance != -1:
            flat_perturbations = filter(lambda p: self.perturbation_distance(text_tokens, p, hypernym_depth) <= distance,
                                         flat_perturbations)

        # candidates generation
        flat_perturbations = list(flat_perturbations)
        perturbed_dfas = list()
        sep, clause_sep = '~~~~~', '|||||'
        for perturbation_tuple in flat_perturbations:
            joined_triple = dfa.to_text(sep=sep, clause_sep=clause_sep)
            perturbation_dic = dict()
            for base_token, perturbation in zip(base_perturbation_tokens, perturbation_tuple):
                perturbation_dic[base_token] = perturbation
                joined_triple = joined_triple.replace(base_token, perturbation)
            clauses = joined_triple.split(clause_sep)
            triples = [clause.split(sep) for clause in clauses]
            if len(perturbation_tuple) > 0:
                perturbed_dfas.append(DFAH(triples, perturbation_dic))

        return perturbed_dfas

    @staticmethod
    def align_attention_matrices(base: numpy.ndarray, misaligned: numpy.ndarray, perturbed_premise: str,
                                 perturbed_hypothesis: str) -> numpy.ndarray:
        """
        Find misalignment between two attention matrices generated by the two pairs (premise, perturbed_premise) and
        hypothesis, perturbed_hypothesis, returning a new tensor based on `misaligned_tensor` such that the two
        have same dimensionality.
        Args:
            base: Base Attention matrix
            misaligned: Attention matrix to align
            perturbed_premise: The perturbed premise
            perturbed_hypothesis: The perturbed hypothesis

        Returns:
            An aligned matrix derived from compressing `misaligned` in the size of `base`.
        """
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        perturbed_text = tokenizer.encode_plus(perturbed_premise, perturbed_hypothesis, add_special_tokens=True, return_tensors='pt')
        perturbed_tokens = [tokenizer.decode(int(i), skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False).replace(' ', '')
                            for i in perturbed_text['input_ids'][0].numpy()]
        perturbed_word_tokens = ['<s>'] + word_tokenize(perturbed_premise) + ['</s>', '</s>'] + \
                                word_tokenize(perturbed_hypothesis) + ['</s>']

        # look for mismatches
        merge_groups = list()
        n, m = len(perturbed_word_tokens), len(perturbed_tokens)
        tokenization_index = 0
        for word_index in range(n):
            if perturbed_word_tokens[word_index] != perturbed_tokens[tokenization_index]:
                # mismatch, start search for complete word
                for k in range(word_index, m):
                    # find sublist with matching word, i.e. the word has been tokenized in subsequent entries
                    if perturbed_word_tokens[word_index] == ''.join(perturbed_tokens[word_index:k]):
                        tokenization_index = k
                        merge_groups.append(list(range(word_index, tokenization_index)))
                        break
            else:
                tokenization_index += 1

        aligned = numpy.ones(base.shape)
        for group in merge_groups:
            if len(group) == 1:
                continue
            misaligned[group[0], :] = numpy.nanmean(misaligned[group, :], axis=0)
            misaligned[:, group[0]] = numpy.nanmean(misaligned[:, group], axis=1)
            misaligned[group[1:], :] = numpy.nan
            misaligned[:, group[1:]] = numpy.nan
        for i in range(misaligned.shape[0]):
            misaligned[i, i] = numpy.inf

        n = aligned.shape[0]
        aligned_row = 0
        for row in misaligned:
            if (~numpy.isnan(row)).sum() == n:
                aligned[aligned_row] = row[~numpy.isnan(row)]
                aligned_row += 1
            else:
                continue
        for k in range(aligned.shape[0]):
            aligned[k, k] = numpy.nan

        return aligned
