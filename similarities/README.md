# Similarity probings
## SBERT
[SBERT](https://arxiv.org/pdf/1908.10084.pdf) is a BERT-based text similarity model.

## InferSent
[InferSent](https://github.com/facebookresearch/InferSent) is an encoding model trained to solve a multitude of tasks.
In this framework, we use it as a semantic text similarity module.

To set it up, run the following (from [InferSent](https://github.com/facebookresearch/InferSent/blob/master/README.md) 's README)):
```shell
# run this from triplex/similarities_modules
git clone https://github.com/facebookresearch/InferSent
mv InferSent infersent # I don't like capital letters my bad
cd infersent

# download embeddings
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
mkdir fastText
curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip fastText/crawl-300d-2M.vec.zip -d fastText/

# download models
mkdir encoder
curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
```