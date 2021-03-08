# TripleX
#### Explaining models, with triples
Triplex is a local explainability method to explain transformer models by creating small knowledge graphs in the form of triplets.
This implementation focuses on explaining predictions on NLI (natural language inference) tasks.
Explanations are provided as `dfas.DFAH` (Deterministic Finite state Automata of Hypernyms).

```python
import pathlib
import copy
import json

from dfas import DFAH

# base path
BASE_PATH = str(pathlib.Path().absolute()) + '/'
# Load a sample DFAH
dfah = DFAH.from_json(BASE_PATH + 'data/dummies/dfah.json')
# Show a DFAH visually
print(dfah)
# access the perturbations it went through
perturbations = dfah.perturbations

# dfah are copy-able and serializable
copy_dfah = copy.copy(dfah)
with open('data/dummies/my_dfah.json') as log:
    json.dump(dfah.to_json(), log)
```

## Getting started
Install dependencies:
```shell
mkvirtualenv triplex # optional, suggested
pip3 install -r requirements.txt
```

### Run
```python
from transformers import AutoModel
import logzero

from triplex import TripleX

# logging level, set to logging.DEBUG for verbose output
logzero.loglevel(logzero.logging.INFO)

model = 'microsoft/deberta-base'
model = AutoModel.from_pretrained(model, output_attentions=True)
# create explainer
explainer = TripleX(model)

premise = 'Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.'
hypothesis = 'Christopher Reeve had an accident.'
dfas, counterfactual_dfas = explainer.extract(premise, hypothesis,
                                              depth=2,
                                              max_perturbations_per_token=3)
print('--- Explanations')
for d in dfas[:3]:
    print(str(d))
for d in counterfactual_dfas[:3]:
    print(str(d))
```

To run on a local JSONL dataset:

```python
from transformers import AutoModel
import pandas as pd

from run_extract import to_standard_labels
from triplex import TripleX

dataset = 'path/to/dataset.jsonl'
data = pd.read_json(dataset, lines=True)
data = data.drop('idx', axis='columns')
data['label'] = to_standard_labels(data['label'].values, dataset)
data = data[['premise', 'hypothesis', 'label']]

model = AutoModel.from_pretrained('microsoft/deberta-base', output_attentions=True)
explainer = TripleX(model)
explanations = list()
for idx, row in data.iterrows():
    premise, hypothesis, label = row.premise, row.hypothesis, row.label
    dfas, counterfactual_dfas = explainer.extract(premise, hypothesis,
                                                  depth=2,
                                                  max_perturbations_per_token=3)
    explanations.append((premise, hypothesis, label, dfas, counterfactual_dfas))
```


### Probing
When analyzing a transformer, it is usually unclear if some heads are more relevant than others for the task at hand.
To understand whether this "head preference" holds for NLI tasks, we leverage the ESNLI dataset to probe the model.
Simply put, we look for the heads with the highest attention weights on the highlighted portions of premise and hypothesis.

You can find scripts to reproduce the analysis in `probings/`.