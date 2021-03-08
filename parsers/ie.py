import itertools
from typing import Tuple, List

from openie import StanfordOpenIE

import pandas
import spacy

from abc import abstractmethod

from dfas import DFA


class Parser:
    @abstractmethod
    def parse(self, data: pandas.DataFrame):
        pass


class OpenIEParser(Parser):
    def __init__(self, port: int = 9000):
        self.doc = spacy.load("en_core_web_sm")
        self.information_extractor = StanfordOpenIE(endpoint='http://localhost:' + str(port))

    def parse(self, text: str) -> DFA:
        triples = self.information_extractor.annotate(text)
        triples = [[triple['subject'], triple['relation'], triple['object']] for triple in triples]

        cleaned_triples_set = self.__clean(triples)  # openie generates redundant and malformed triples
        dfas = DFA(cleaned_triples_set, text)

        return dfas

    def __clean(self, triples: List[List]) -> List[Tuple[str, str, str]]:
        """
        Clean the given triples:
            - remove nested repetitions:
                S - P - advances
                S - P - tech advances
                S - P - new tech advances
                becomes
                S - P - new tech advances
                advances - is - new
                advances - is - tech
            - minimal predicates
                S - show promise - for future research
                becomes
                S - show - promise
                promise - ? - for future research
            - remove triples with non-verb predicates
            - lemmatize
            - remove auxiliaries
            - move adjectives into new triples (noun, 'is', adj)
            - move adverbs into new triples (verb, 'ADV', adv)
            - remove particles (articles, preposition)

        Args:
            triples: The triples

        Returns:
            The cleaned triples.
        """
        cleaned_triples = list()
        additional_cleaned_triples = list()
        current_triples = list()

        for i, (s, p, o) in enumerate(triples):
            triple_doc = self.doc(' '.join([s, p, o]))
            subject_doc = self.doc(triples[i][0])
            object_doc = self.doc(triples[i][2])
            predicate_doc = triple_doc[len(subject_doc):len(triple_doc) - len(object_doc)]

            # remove triples with non-verb predicates
            parts_of_speech = [el.pos_ for el in predicate_doc]
            if 'VERB' not in parts_of_speech and 'AUX' not in parts_of_speech:
                continue

            # keep track of what token is where to avoid replacing the wrong one, if repeated across
            # subject, predicate or object
            s_doc_len, p_doc_len, o_doc_len = len(subject_doc), len(predicate_doc), len(object_doc)
            # lemmatize common nouns
            for t, token in enumerate(triple_doc):
                # TODO: Check if no lemmatization improves
                # if token.pos_ != 'PROPN':
                #     if t < s_doc_len:
                #         triples[i][0] = triples[i][0].replace(token.text, token.lemma_.lower())
                #     elif s_doc_len <= t < s_doc_len + p_doc_len:
                #         triples[i][1] = triples[i][1].replace(token.text, token.lemma_.lower())
                #     else:
                #         triples[i][2] = triples[i][2].replace(token.text, token.lemma_.lower())
                # TODO
                # remove adjectives, move them to new triples
                if token.pos_ == 'ADJ':
                    head = token.head.text
                    additional_cleaned_triples.append((head, 'is', token.lemma_))
                    # remove from triple
                    if t < s_doc_len:
                        triples[i][0] = triples[i][0].replace(token.text, '')
                    elif s_doc_len <= t < s_doc_len + p_doc_len:
                        triples[i][1] = triples[i][1].replace(token.text, '')
                    else:
                        triples[i][2] = triples[i][2].replace(token.text, '')
                # remove adverbs, move them to new triples
                elif token.pos_ == 'ADV':
                    # spacy bug on multiple adverbs which create a chain of ADV
                    if token.head.pos_ == 'ADV':
                        cur_head = token.head
                        chain = list()
                        while cur_head.pos_ == 'ADV':
                            # spacy bug: recursive cycles
                            if cur_head.text == token.text or cur_head.text in chain:
                                break
                            chain.append(cur_head.text)
                            cur_head = cur_head.head
                        if isinstance(cur_head, str) and cur_head != 'ADV':
                            additional_cleaned_triples.append((cur_head.lower(), 'ADV', token.lemma_))
                    else:
                        head = token.head.text
                        if isinstance(head, str):
                            additional_cleaned_triples.append((head.lower(), 'ADV', token.lemma_))
                    # remove from triple
                    if t < s_doc_len:
                        triples[i][0] = triples[i][0].replace(token.text, '')
                    elif s_doc_len <= t < s_doc_len + p_doc_len:
                        triples[i][1] = triples[i][1].replace(token.text, '')
                    else:
                        triples[i][2] = triples[i][2].replace(token.text, '')
                elif token.pos_ in ['ADP', 'DET']:
                    # remove from triple
                    if t < s_doc_len:
                        triples[i][0] = triples[i][0].replace(token.text, '')
                    elif s_doc_len <= t < s_doc_len + p_doc_len:
                        triples[i][1] = triples[i][1].replace(token.text, '')
                    else:
                        triples[i][2] = triples[i][2].replace(token.text, '')

            # predicate cleaning
            triple_doc = self.doc(' '.join([triples[i][0], triples[i][1], triples[i][2]]))
            subject_doc = self.doc(triples[i][0])
            object_doc = self.doc(triples[i][2])
            predicate_doc = triple_doc[len(subject_doc):len(triple_doc) - len(object_doc)]
            predicate_parts_of_speech = [el.pos_ for el in predicate_doc]
            if 'VERB' in predicate_parts_of_speech:
                for token in predicate_doc:
                    # remove auxiliaries for predicates with a verb
                    if token.pos_ == 'AUX':
                        triples[i][1] = triples[i][1].replace(token.text, '')

            # spacy bug: when alone, the predicate may be reevaluated as another POS
            if 'VERB' in predicate_parts_of_speech:
                verb_tag = 'VERB'
            elif 'AUX' in predicate_parts_of_speech:
                # sometimes the verb is considered an auxiliary verb
                verb_tag = 'AUX'
            elif len(predicate_parts_of_speech) == 1 or\
                    len(predicate_parts_of_speech) == 2 and 'SPACE' in predicate_parts_of_speech:
                verb_tag = predicate_parts_of_speech[0] if predicate_parts_of_speech[0] != 'SPACE'\
                    else predicate_parts_of_speech[1]
            else:
                continue
            # move non-verb tokens to the subject (before verb)
            triples[i][0] += ' ' + ' '.join(el.text for el in predicate_doc[:predicate_parts_of_speech.index(verb_tag) + 1]
                                            if el.pos_ not in ['AUX', 'VERB'])
            # move non-verb tokens to the object (after verb)
            triples[i][2] = ' ' + ' '.join(el.text for el in predicate_doc[predicate_parts_of_speech.index(verb_tag) + 1:]
                                           if el.pos_ not in ['AUX', 'VERB']) + triples[i][2]
            triples[i][1] = ' ' + ' '.join(el.text for el in predicate_doc if el.pos_ == verb_tag)

            current_triples.append(triples[i])

        # remove nested repetitions by picking the largest encompassing
        # first grouping: by predicate
        triples_groups = sorted(current_triples, key=lambda x: x[1])
        triples_groups = itertools.groupby(triples_groups, lambda x: x[1])
        triples_groups = [(predicate, list(predicate_group)) for predicate, predicate_group in triples_groups]
        # second grouping: by object (first word), since the same predicate can have multiple objects
        current_triplets = list()
        for predicate, predicate_group in triples_groups:
            if len(predicate_group) == 1:
                current_triplets.append(predicate_group)
                continue

            subjects = [self.doc(s) for s, _, _ in predicate_group]
            objects = [self.doc(o) for _, _, o in predicate_group]
            shortest_object_prefix_len = min(len(o) for o in objects)
            shortest_subject_suffix_len = min(len(s) for s in subjects)
            grouping_anchors = [(s[-shortest_subject_suffix_len:].text + ' ~~~~~~~~~~~ '
                                 + o[:-shortest_object_prefix_len].text, s.text, o.text)
                                for s, o in zip(subjects, objects)]
            grouping_anchors = sorted(grouping_anchors, key=lambda x: x[0])
            groups = itertools.groupby(grouping_anchors, lambda x: x[0])
            groups = [list(group) for _, group in groups]

            # add only largest triple from each group
            for group in groups:
                group_anchor = max(group, key=lambda x: len(x[1].split(' ')) + len(x[2].split(' ')))
                current_triplets.append([group_anchor[1].lower(), predicate, group_anchor[2].lower()])

        # make tuples out of lists
        for s, p, o in current_triples:
            if len(s) > 0 and len(p) > 0 and len(o) > 0:
                cleaned_triples.append((s.strip(), p.strip(), o.strip()))
        cleaned_triples = cleaned_triples + additional_cleaned_triples

        return cleaned_triples
