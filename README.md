# morph-tok-eval

## Replicating the paper experiments

Install the dependencies using `requirements.txt`.

This repository uses [Snakemake](https://snakemake.readthedocs.io/en), a Python
extension of Makefile, to manage the experiments.

You can run the all the experiments (including downloading CC100 data and
training the tokenizers) by running:

```bash
snakemake --executor local --cores all
```

This will run all computations locally while using all available CPU cores.
Snakemake also support paralleliziation in cluster environment. For this, you
might need to adjust the specifications of computational resources of the
Snakemake rules to match your cluster specifics.

The Snakemake steps are:

* Download CC100 datasets for the studied languages (rule `download_cc100`).

* Train BPE, WordPiece and Unigram tokenizers using the Huggingface tokenizers
  library (rule `train_tokenizer`).

* Tokenize word lists using the trained tokenizers (rule
  `tokenize_unimorph_our_tokenizer`) and with character-level (rule
  `character_tokenization`) segmentation and gold segmentation (rule
  `gold_tokenization`).

* Evaluate all tokenizations using boundary precision and recall and different
  variants of the proposed metric (rule `evaluate_segmentation`).

* Compute the correlations between the alignment metric and precision boundary
  and recall (rule `compute_correlations`).

After running all experiments, the results are in directory `correlations`.
