import spacy

import itertools
import re
from abc import abstractmethod
from typing import List, Tuple

from dfas import DFA

DOC = spacy.load("en_core_web_sm")


class Parser:
    @abstractmethod
    def parse(self, text: str) -> DFA:
        pass


def clean(triples: List[List]) -> List[Tuple[str, str, str]]:
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
        - remove duplicates
        - removes included triples
        - remove triples with unclear subjects (pronouns)

    Args:
        triples: The triples

    Returns:
        A list of pairs (cleaned triples, lemmatization dic)
    """
    cleaned_triples = list()
    current_triples = list()
    # remove triplets with any of these subjects
    banned_subjects = ['i', 'you', 'he', 'she', 'it', 'we', 'they']

    for i, (s, p, o) in enumerate(triples):
        lemmatization_dic = dict()
        triple_doc = DOC(' '.join([s, p, o]))
        subject_doc = DOC(triples[i][0])
        object_doc = DOC(triples[i][2])
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
            if token.pos_ != 'PROPN':
                if t < s_doc_len:
                    triples[i][0] = triples[i][0].replace(token.text, token.lemma_.lower())
                # elif s_doc_len <= t < s_doc_len + p_doc_len:
                #     triples[i][1] = triples[i][1].replace(token.text, token.lemma_.lower())
                else:
                    triples[i][2] = triples[i][2].replace(token.text, token.lemma_.lower())
                lemmatization_dic[token.text] = token.lemma_.lower()
            # remove adjectives, move them to new triples
            # if token.pos_ == 'ADJ':
                # additional_cleaned_triples.append((head, 'is', token.lemma_))
                # remove from triple
                # if t < s_doc_len:
                #     triples[i][0] = triples[i][0].replace(token.text, '')
                # elif s_doc_len <= t < s_doc_len + p_doc_len:
                #     triples[i][1] = triples[i][1].replace(token.text, '')
                # else:
                #     triples[i][2] = triples[i][2].replace(token.text, '')
            # move adverbs to new triples
            elif token.pos_ == 'ADV':
                if t < s_doc_len:
                    triples[i][0] = triples[i][0].replace(token.text, '')
                elif s_doc_len <= t < s_doc_len + p_doc_len:
                    triples[i][1] = triples[i][1].replace(token.text, '')
                else:
                    triples[i][2] = triples[i][2].replace(token.text, '')

        # predicate cleaning
        triple_doc = DOC(' '.join([triples[i][0], triples[i][1], triples[i][2]]))
        subject_doc = DOC(triples[i][0])
        object_doc = DOC(triples[i][2])
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
        elif len(predicate_parts_of_speech) == 1 or \
                len(predicate_parts_of_speech) == 2 and 'SPACE' in predicate_parts_of_speech:
            verb_tag = predicate_parts_of_speech[0] if predicate_parts_of_speech[0] != 'SPACE' \
                else predicate_parts_of_speech[1]
        else:
            continue
        # move non-verb tokens to the subject (before verb)
        triples[i][0] += ' ' + ' '.join(el.text for el in predicate_doc[:predicate_parts_of_speech.index(verb_tag) + 1]
                                        if el.pos_ not in ['AUX', 'VERB'])
        # move non-verb tokens to the object (after verb)
        triples[i][2] = ' ' + ' '.join(el.text for el in predicate_doc[predicate_parts_of_speech.index(verb_tag) + 1:]
                                       if el.pos_ not in ['AUX', 'VERB']) + ' ' + triples[i][2]
        triples[i][1] = ' ' + ' '.join(el.text for el in predicate_doc if el.pos_ == verb_tag)

        current_triples.append([re.sub(' {2,}', ' ', triples[i][0]).strip(),
                                re.sub(' {2,}', ' ', triples[i][1]).strip(),
                                re.sub(' {2,}', ' ', triples[i][2]).strip(),
                                lemmatization_dic])

    # remove nested repetitions by picking the largest encompassing
    # first grouping: by predicate
    triples_groups = sorted(current_triples, key=lambda x: x[1])
    triples_groups = itertools.groupby(triples_groups, lambda x: x[1])
    triples_groups = [(predicate, list(predicate_group)) for predicate, predicate_group in triples_groups]
    # second grouping: by object (first word), since the same predicate can have multiple objects
    current_triples = list()
    for predicate, predicate_group in triples_groups:
        if len(predicate_group) == 1:
            current_triples.append(predicate_group[0])
            continue

        subjects = [DOC(s) for s, _, _, _ in predicate_group]
        objects = [DOC(o) for _, _, o, _ in predicate_group]
        lemmatization_dic = [d for _, _, _, d in predicate_group]
        shortest_object_prefix_len = min(len(o) for o in objects)
        shortest_subject_suffix_len = min(len(s) for s in subjects)
        grouping_anchors = [(s[-shortest_subject_suffix_len:].text + '~~~' + o[:shortest_object_prefix_len].text,
                             s.text, o.text, d) for s, o, d in zip(subjects, objects, lemmatization_dic)]
        grouping_anchors = sorted(grouping_anchors, key=lambda x: x[0])
        groups = itertools.groupby(grouping_anchors, lambda x: x[0])
        groups = [list(group) for _, group in groups]

        # add only largest triple from each group
        for group in groups:
            group_anchor = max(group, key=lambda x: len(x[1].split(' ')) + len(x[2].split(' ')))
            current_triples.append([group_anchor[1], predicate, group_anchor[2], group_anchor[3]])

    # some object groups miss starting tokens
    n = len(current_triples)
    current_triples_buffer = list()
    for i in range(n):
        subj, pred, obj, d = current_triples[i]
        subset_index = [obj in other_triple[2] and subj in other_triple[0] and pred == other_triple[1]
                        if i != j else False for j, other_triple in enumerate(current_triples)]
        # this triple is not subset of any another, preserve it
        if not any(subset_index):
            current_triples_buffer.append(current_triples[i])

    # make tuples out of lists
    for s, p, o, d in current_triples_buffer:
        if len(s) > 0 and len(p) > 0 and len(o) > 0:
            if s.strip() not in banned_subjects:
                cleaned_triples.append([s.strip(), p.strip(), o.strip(), d])

    for i in range(len(cleaned_triples)):
        lemmatization_dic = cleaned_triples[i][3]
        for original, lemmatized in lemmatization_dic.items():
            cleaned_triples[i][0] = cleaned_triples[i][0].replace(lemmatized, original)
            # cleaned_triples[i][1] = cleaned_triples[i][1].replace(lemmatized, original)
            cleaned_triples[i][2] = cleaned_triples[i][2].replace(lemmatized, original)
    cleaned_triples = [tuple([s, p, o]) for s, p, o, _ in cleaned_triples]

    return cleaned_triples
