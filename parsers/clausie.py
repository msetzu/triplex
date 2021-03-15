from pyclausie import ClausIE as ClausIEpy

from dfas import DFA
from parsers.utils import Parser, clean


class ClausIE(Parser):
    def __init__(self):
        self.parser = ClausIEpy.get_instance()

    def parse(self, text: str) -> DFA:
        triples = self.parser.extract_triples([text])
        triples = [[t['subject'], t['predicate'], t['object']] for t in triples]
        triples = clean(triples)
        dfa = DFA(triples, text)

        return dfa
