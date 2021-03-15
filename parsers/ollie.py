import os

import time
import subprocess
import pathlib

from dfas import DFA
from parsers.utils import Parser


class Ollie(Parser):
    def __init__(self):
        self.path = str(pathlib.Path(__file__).parent.absolute()) + '/'
        self.ollie_path = str(pathlib.Path(__file__).parent.absolute()) + '/../parsers_modules/ollie/'

    def parse(self, text: str) -> DFA:
        input_file = self.path + '.' + str(time.time())
        with open(input_file, 'w') as log:
            log.write(text)

        # run ollie
        os.chdir(self.ollie_path)
        output = subprocess.run(["java", "-Xmx512m", "-jar", "ollie-app-latest.jar", input_file], capture_output=True)
        os.chdir(self.path)
        triples = str(output.stdout).split('\\n')[1:-2]
        for i in range(len(triples)):
            triples[i] = triples[i].split(': (')[1:][0].split('; ')
            # object also includes a closing ')'
            triples[i][-1] = triples[i][-1][:-1]
            triples[i] = tuple(triples[i])

        os.remove(input_file)
        dfa = DFA(triples, text)

        return dfa



