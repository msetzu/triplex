from typing import Union

import fire
import spacy
import pandas
import numpy
from tqdm import tqdm

import pathlib

from transformers import AutoModel, RobertaTokenizer
import torch
from pandas import DataFrame


DATA_FOLDER = str(pathlib.Path().absolute()) + '/../data/esnli/'
nlp = spacy.load("en_core_web_sm")


def attention_highlights(model, tokenizer, premise: str, hypothesis: str, highlights_idxs: list):
    """Compute attention scores of `model` on `text`, return rows and columns given by `highlight_idxs` ."""
    x = tokenizer.encode_plus(premise, hypothesis, add_special_tokens=True, return_tensors='pt')
    attentions = model(x['input_ids'])[-1]
    # Remove first dimension for single prediction
    attentions = torch.cat(attentions, dim=0)
    attentions = attentions[-1].detach().numpy()
    # attention pairs on each head
    highlighted_attentions = [attentions[:, highlights_idxs, :], attentions[:, :, highlights_idxs]]

    return highlighted_attentions


def head_values(model: str = 'microsoft/deberta-base'):
    null_highlights = {'{}'}
    esnli = pandas.read_csv(DATA_FOLDER + 'esnli_dev.csv')

    model = AutoModel.from_pretrained(model, output_attentions=True)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    nr_heads = model.base_model.config.num_attention_heads

    highlights_per_annotation = list()
    for row in tqdm(esnli.itertuples(), total=esnli.shape[0]):
        highlights = list()
        premise = row.Sentence1
        hypothesis = row.Sentence2
        annotations_idxs = [(row.Sentence1_marked_1, row.Sentence2_marked_1),
                            (row.Sentence1_marked_2, row.Sentence2_marked_2),
                            (row.Sentence1_marked_3, row.Sentence2_marked_3)]

        for premise_highlight, hypothesis_highlight in annotations_idxs:
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

                attention_vectors = attention_highlights(model, tokenizer, premise, hypothesis, highlight_tokenizer_idx)
                highlights.append(attention_vectors)

        highlights_per_annotation.append((premise, hypothesis, highlights) if len(highlights) > 0 else None)

    return highlights_per_annotation, nr_heads


def head_rank(attention_vectors, nr_heads):
    heads_scores = list()
    for highlight in attention_vectors:
        summary_vector = numpy.zeros(nr_heads,)
        for row, col in highlight:
            summary_vector += (row.reshape(col.shape) + col).mean(axis=(1, 2))
        heads_scores.append(summary_vector / len(attention_vectors))
    heads_scores = numpy.array(heads_scores)
    heads_ranks = heads_scores.argsort(axis=1)

    mean_ranks = heads_ranks.mean(axis=0).reshape(nr_heads, 1)
    std_ranks = heads_ranks.std(axis=0).reshape(nr_heads, 1)
    min_ranks = heads_ranks.sum(axis=0).argsort().reshape(nr_heads, 1)
    median_ranks = numpy.median(heads_ranks, axis=0).reshape(nr_heads, 1)
    ranks = numpy.concatenate([mean_ranks, std_ranks, min_ranks, median_ranks], axis=1)
    ranks = DataFrame(ranks, columns=['mean_rank', 'std_ranks', 'min_rank', 'median_rank'])

    return ranks


def main(model: str = 'microsoft/deberta-base', out: Union[str, None] = None):
    highlights, nr_heads = head_values(model)
    rank_df = head_rank([h[2] for h in highlights if h is not None], nr_heads)
    if out is not None:
        rank_df.to_csv(out, index=False)


if __name__ == '__main__':
    fire.Fire(main)
