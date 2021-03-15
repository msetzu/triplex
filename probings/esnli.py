from typing import Union

import fire
import spacy
import pandas
import numpy
from tqdm import tqdm

import pathlib

from transformers import AutoModel, RobertaTokenizer
import torch


DATA_FOLDER = str(pathlib.Path().absolute()) + '/../data/esnli/'
nlp = spacy.load("en_core_web_sm")


def attention_highlights(model, tokenizer, premise: str, hypothesis: str, highlights_idxs: list):
    """Compute attention scores of `model` on `text` on layer `layer`, return rows and columns given by `highlight_idxs` ."""
    x = tokenizer.encode_plus(premise, hypothesis, add_special_tokens=True, return_tensors='pt')
    attentions = model(x['input_ids'])[-1]
    # Remove first dimension for single prediction
    attentions = torch.cat(attentions, dim=0)

    highlighted_attentions = [attentions[layer].detach().numpy() for layer in range(attentions.size()[0])]
    highlighted_attentions = [[layer, attentions[:, highlights_idxs, :], attentions[:, :, highlights_idxs]]
                              for layer, attentions in enumerate(highlighted_attentions)]

    return highlighted_attentions


def head_values(model):
    null_highlights = {'{}'}
    esnli = pandas.read_csv(DATA_FOLDER + 'esnli_dev.csv').iloc[:100]
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')

    highlights_per_annotation = list()
    for row in tqdm(esnli.itertuples(), total=esnli.shape[0]):
        row_highlights = list()
        premise = row.Sentence1
        hypothesis = row.Sentence2
        annotations_idxs = [(row.Sentence1_marked_1, row.Sentence2_marked_1),
                            (row.Sentence1_marked_2, row.Sentence2_marked_2),
                            (row.Sentence1_marked_3, row.Sentence2_marked_3)]

        for annotation_idx, (premise_highlight, hypothesis_highlight) in enumerate(annotations_idxs):
            if premise_highlight in null_highlights or hypothesis_highlight in null_highlights:
                continue
            else:
                tokens = tokenizer.encode_plus(premise, hypothesis, add_special_tokens=True, return_tensors='pt')['input_ids'][0].numpy()
                tokens = [tokenizer.decode(int(i), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).replace(' ', '')
                          for i in tokens]
                separator_token_idx = tokens.index('</s>')
                premise_tokens, hypothesis_tokens = tokens[:separator_token_idx], tokens[separator_token_idx:]
                n = len(tokens)
                highlighted_tokens_premise = [('premise', tokens)
                                              for i, tokens in enumerate(premise_highlight.split('*')) if i % 2 != 0]
                highlighted_tokens_hypothesis = [('hypothesis', tokens) for i, tokens in enumerate(hypothesis_highlight.split('*'))
                                                 if i % 2 != 0]
                highlight_tokenizer_idx = list()
                for source, h in highlighted_tokens_premise + highlighted_tokens_hypothesis:
                    # exact match
                    if source == 'premise' and h in premise_tokens:
                        # tokens can be repeated in premise and hypothesis, separate them
                        highlight_tokenizer_idx.append(tokens.index(h))
                    elif source == 'hypothesis' and h in hypothesis_tokens:
                        highlight_tokenizer_idx.append(hypothesis_tokens.index(h) + separator_token_idx)
                    else:
                        # a word has been split into multiple tokens, re-align it
                        highlight_len = len(h)
                        tokens_splits = [[[h[:a], h[a:]] for a in range(highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:]) > 0))],
                                         [[h[:a], h[a:b], h[b:]] for a in range(highlight_len)
                                          for b in range(a, highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:b]) > 0, len(h[b:]) > 0))],
                                         [[h[:a], h[a:b], h[b:c], h[c:]] for a in range(highlight_len)
                                          for b in range(a, highlight_len) for c in range(b, highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:b]) > 0, len(h[b:c]) > 0, len(h[c:]) > 0))],
                                         [[h[:a], h[a:b], h[b:c], h[c:d], h[d:]] for a in range(highlight_len)
                                          for b in range(a, highlight_len) for c in range(b, highlight_len)
                                          for d in range(c, highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:b]) > 0, len(h[b:c]) > 0, len(h[c:d]) > 0,
                                                  len(h[d:]) > 0))],
                                         [[h[:a], h[a:b], h[b:c], h[c:d], h[d:e], h[e:]] for a in range(highlight_len)
                                          for b in range(a, highlight_len) for c in range(b, highlight_len)
                                          for d in range(c, highlight_len) for e in range(d, highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:b]) > 0, len(h[b:c]) > 0, len(h[c:d]) > 0,
                                                  len(h[d:e]) > 0, len(h[e:]) > 0))],
                                         [[h[:a], h[a:b], h[b:c], h[c:d], h[d:e], h[e:f], h[f:]] for a in range(highlight_len)
                                          for b in range(a, highlight_len) for c in range(b, highlight_len)
                                          for d in range(c, highlight_len) for e in range(d, highlight_len)
                                          for f in range(highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:b]) > 0, len(h[b:c]) > 0, len(h[c:d]) > 0,
                                                  len(h[d:e]) > 0, len(h[e:f]) > 0, len(h[f:]) > 0))],
                                         [[h[:a], h[a:b], h[b:c], h[c:d], h[d:e], h[e:f], h[f:g], h[g:]]
                                          for a in range(highlight_len) for b in range(a, highlight_len)
                                          for c in range(b, highlight_len) for d in range(c, highlight_len)
                                          for e in range(d, highlight_len) for f in range(highlight_len)
                                          for g in range(highlight_len)
                                          if all((len(h[:a]) > 0, len(h[a:b]) > 0, len(h[b:c]) > 0, len(h[c:d]) > 0,
                                                  len(h[d:e]) > 0, len(h[e:f]) > 0, len(h[f:g]) > 0, len(h[g:]) > 0))]
                                         ]
                        found = False
                        for split_length in range(2, 9):
                            splits = tokens_splits[split_length - 2]
                            matches = [[tokens[i:i + split_length] == s for i in range(n - split_length + 1)]
                                       for s in splits]
                            match_index = [match.index(True) for match in matches if True in match]
                            if len(match_index) > 0:
                                highlight_tokenizer_idx = highlight_tokenizer_idx + \
                                                          list(range(match_index[0], match_index[0] + split_length))
                                found = True
                                break
                        if not found:
                            # match not found
                            raise ValueError('No match for ' + h + ' in ' + str(tokens))

                # retrieve attention vectors for the highlights only  (highlight_tokenized_idx)
                attention_vectors = attention_highlights(model, tokenizer, premise, hypothesis, highlight_tokenizer_idx)
                for annotation_attention_vector in attention_vectors:
                    row_highlights.append(annotation_attention_vector)

        highlights_per_annotation.append((premise, hypothesis, row_highlights) if len(row_highlights) > 0
                                         else (premise, hypothesis, None))

    return highlights_per_annotation


def head_rank(attention_vectors, nr_heads: int, nr_layers: int):
    alignments = list()
    for sample in attention_vectors:
        annotation_vectors = [sample[i * nr_layers: (i + 1) * nr_layers] for i in range(len(sample) // nr_layers)]
        for annotation in annotation_vectors:
            annotation_alignments = numpy.array([(layer_attentions[1].reshape(layer_attentions[2].shape) +
                                                  layer_attentions[2]).mean(axis=(1, 2))
                                                 for layer_attentions in annotation])
            alignments.append(numpy.flip(annotation_alignments.argsort(axis=None)))

    alignments = numpy.array(alignments)
    data = list()
    for head in range(nr_layers * nr_heads):
        head_ranks = numpy.where(alignments == head)[1]
        mean_rank, std_rank = head_ranks.mean(), head_ranks.std()
        layer_nr, head_nr = numpy.unravel_index(head, (nr_layers, nr_heads))
        data.append((head, layer_nr.item(), head_nr.item(), mean_rank, std_rank))
    data = sorted(data, key=lambda x: x[3])
    order = list(range(nr_layers * nr_heads))

    ranks = pandas.DataFrame(data, columns=['head_idx', 'layer', 'head', 'mean_rank', 'std_rank'])
    ranks['rank'] = order

    return ranks


def main(model: str = 'microsoft/deberta-base', out: Union[str, None] = None):
    model_name = model
    model = AutoModel.from_pretrained(model, output_attentions=True)
    model.training = False
    nr_heads = model.base_model.config.num_attention_heads
    nr_layers = model.base_model.config.num_hidden_layers

    highlights = head_values(model)
    ranks = head_rank([attention_vectors for premise, hypothesis, attention_vectors in highlights
                       if attention_vectors is not None], nr_heads, nr_layers)

    ranks['model_name'] = model_name
    ranks = ranks[['model_name', 'layer', 'head', 'rank', 'mean_rank', 'std_rank', 'head_idx']]
    if out is not None:
        ranks.to_csv(out, index=False)


if __name__ == '__main__':
    fire.Fire(main)
