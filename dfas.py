from typing import Union, Tuple, List

import json


class State:
    """
    A DFAX state.
    """
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return 'State [{0}]'.format(self.name)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        """
        Create a new state, by append `other` to this state's name.
        Args:
            other (Union[str, State]): The summing string/state

        Returns:
            (State): A new state with the appended `other`
        """
        if isinstance(other, State):
            return State(self.name + ' ' + other.name)
        if isinstance(other, str):
            self.name += ' ' + other

    def __len__(self):
        return len(self.name)


class Transition:
    """
    A DFAX transition.
    """
    def __init__(self, source: State, transition_name: str, destination: State):
        self.source = source
        self.destination = destination
        self.name = transition_name
        self.transition = (self.destination, self.name, self.source)

    def __str__(self):
        return '{0} --- {1} ---> {2}'.format(str(self.source), self.name, str(self.destination))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Transition) and self.destination == other.destination and \
               self.source == other.source and self.transition == other.transition

    def __hash__(self):
        return hash((self.source.name + ' ' + self.name + ' ' + self.destination.name))


class DFA:
    """
    A Deterministic Finite state Automaton for eXplanation.
    """
    def __init__(self, triples: Union[List[Tuple[str, str, str]], Tuple[str, str, str]]):
        """
        Create a DFAX from a (set of) tuples (state, transition, state)
        Args:
            triples: The triple(s) defining the transitions
        """
        self.triples_list = triples if isinstance(triples, list) else [triples]
        states = [s for s, _, _ in self.triples_list] + [o for _, _, o in self.triples_list]

        # wrapped names
        self.states_dic = {state_name: State(state_name) for state_name in states}
        self.transitions_dic = {(s, p, o): Transition(self.states_dic[s], p, self.states_dic[o])
                                for s, p, o in self.triples_list}
        self.triples_dic = {(s, p, o): (self.states_dic[s],
                                        self.transitions_dic[(s, p, o)],
                                        self.states_dic[o])
                            for s, p, o in self.triples_list}
        self.states = set(self.states_dic.values())
        self.transitions = [self.transitions_dic[(s, p, o)] for s, p, o in self.triples_list]

        # names
        self._states_names = states

    def __str__(self):
        out = '--- States\n'
        for state in self.states:
            out += '\t' + str(state) + ' | '
        out += '\n--- Transitions\n'
        for transition in self.transitions:
            out += '\t' + str(transition) + ' \n'

        return out

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, DFA):
            return False

        return self.triples_list == other.triples_list and self.triples_list == other.triples_list

    def __copy__(self):
        transitions = [(s.name, p.name, o.name) for s, p, o in self.transitions_dic]
        return DFA(transitions)

    def triples(self) -> List[Tuple[str, str, str]]:
        """Return this DFA as a tuple (subject, predicate, object)"""
        return [(self.triples_dic[(s, p, o)][0].name,
                 self.triples_dic[(s, p, o)][1].name,
                 self.triples_dic[(s, p, o)][2].name)
                for (s, p, o) in self.triples_list]

    def to_json(self) -> dict:
        """Return a JSON representation of this object"""
        return {
            'triples':
                [[self.triples_dic[(s, p, o)][0].name,
                  self.triples_dic[(s, p, o)][1].name,
                  self.triples_dic[(s, p, o)][2].name]
                 for (s, p, o) in self.triples_list],
        }

    @staticmethod
    def from_json(json_file: str, jsonl: bool = False):
        """
        Read a json-stored DFAX. Set `jsonl` to True to read a list of
        DFAX stored in a JSONL
        Args:
            json_file: Path to the file.
            jsonl: True to read multiple DFAXs from a JSONL file, False otherwise.
                    Defaults to False.

        Returns:
            The read DFAX(s).
        """
        with open(json_file, 'r') as log:
            if not jsonl:
                triples = json.load(log)
                return DFA([(str(s), str(p), str(o)) for [s, p, o] in triples])
            else:
                dfas = list()
                for line in log:
                    triples = json.loads(line)
                    dfa = DFA([(str(s), str(p), str(o)) for [s, p, o] in triples])
                    dfas.append(dfa)
                return dfas

    def to_text(self, sep: str = ' ', clause_sep: str = '. ') -> str:
        """
        Return a textual representation of this DFAX to feed a model.
        Joins single clauses of a triple with `sep`, then joins triples with `clause_sep`.
        Args:
            sep: The triple separator. Defaults to ' '
            clause_sep: The triples separator. Defaults to '. '

        Returns:

        """
        return clause_sep.join(sep.join(triple) for triple in self.triples())


class DFAH(DFA):
    """
    A Deterministic Finite state Automaton for eXplanation.
    """
    def __init__(self, triples: Union[List[Tuple[str, str, str]], Tuple[str, str, str]],
                 perturbations: Union[None, dict, List[Tuple[str, str]]] = None):
        """
        Create a DFAX from a (set of) tuples (state, transition, state)
        Args:
            triples: The triple(s) defining the transitions
            perturbations: Perturbations applied to the states of this DFA, if any
        """
        super().__init__(triples)
        if isinstance(perturbations, dict):
            self.perturbations = perturbations
        elif isinstance(perturbations, tuple):
            self.perturbations = dict(perturbations)
        elif perturbations is None:
            self.perturbations = dict()

    def __str__(self):
        out = super().__str__()
        if self.perturbations is not None:
            out += '\n--- Perturbations:'
            for original, perturbation in self.perturbations.items():
                print('\t{0} => {1}'.format(original, perturbation))

        return out

    def __eq__(self, other):
        if not isinstance(other, DFAH):
            return False

        return self.triples_list == other.triples_list and self.triples_list == other.triples_list and\
               self.perturbations == other.perturbations

    def __copy__(self):
        transitions = [(s.name, p.name, o.name) for s, p, o in self.transitions_dic]
        perturbations = self.perturbations
        return DFAH(transitions, perturbations)

    def triples(self) -> List[Tuple[str, str, str]]:
        """Return this DFA as a tuple (subject, predicate, object)"""
        return [(self.triples_dic[(s, p, o)][0].name,
                 self.triples_dic[(s, p, o)][1].name,
                 self.triples_dic[(s, p, o)][2].name)
                for (s, p, o) in self.triples_list]

    def to_json(self) -> dict:
        """Return a JSON representation of this object"""
        return {
            'triples':
                [[self.triples_dic[(s, p, o)][0].name,
                  self.triples_dic[(s, p, o)][1].name,
                  self.triples_dic[(s, p, o)][2].name]
                 for (s, p, o) in self.triples_list],
            'perturbations': self.perturbations
        }

    @staticmethod
    def from_json(json_file: str, jsonl: bool = False):
        """
        Read a json-stored DFAX. Set `jsonl` to True to read a list of
        DFAX stored in a JSONL
        Args:
            json_file: Path to the file.
            jsonl: True to read multiple DFAXs from a JSONL file, False otherwise.
                    Defaults to False.

        Returns:
            The read DFAX(s).
        """
        with open(json_file, 'r') as log:
            if not jsonl:
                dfa = json.load(log)
                return DFAH([(str(s), str(p), str(o)) for [s, p, o] in dfa['triples']], dfa.get('perturbations', dict()))
            else:
                dfas = list()
                for line in log:
                    dfa = json.loads(line)
                    dfa = DFAH([(str(s), str(p), str(o)) for [s, p, o] in dfa], dfa.get('perturbations', dict()))
                    dfas.append(dfa)
                return dfas
