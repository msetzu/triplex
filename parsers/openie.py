from openie import StanfordOpenIE

import spacy

from dfas import DFA
from parsers.utils import Parser, clean


class OpenIE(Parser):
    def __init__(self, port: int = 9000):
        self.doc = spacy.load("en_core_web_sm")
        self.information_extractor = StanfordOpenIE(endpoint='http://localhost:' + str(port))

    def parse(self, text: str) -> DFA:
        triples = self.information_extractor.annotate(text, properties={'relationsBeam': 1.0})
        triples = [[triple['subject'], triple['relation'], triple['object']] for triple in triples]

        cleaned_triples = clean(triples)
        dfa = DFA(cleaned_triples, text)

        return dfa
