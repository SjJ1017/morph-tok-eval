#!/usr/bin/env python3

"""Train a POS tagger on UD data with different tokenization schemes."""

from typing import Dict, IO, List, Tuple
import argparse
from collections import defaultdict
import hashlib
import logging
import os
import pickle
import random
import sys

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import PreTrainedTokenizerFast
from gensim.models.fasttext import FastText
import numpy as np

# Add Legros Python path and import Legros modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'legros', 'python'))
from legros.vocab import Vocab

# Set enviroment variable TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

UD_HOME = "/lnet/ms/data/universal-dependencies-2.16"

CODE_TO_NAME = {
    "ces": "UD_Czech-PDTC/cs_pdtc-ud",
    "deu": "UD_German-HDT/de_hdt-ud",
    "eng": "UD_English-EWT/en_ewt-ud",
    "fin": "UD_Finnish-TDT/fi_tdt-ud",
    "hbs": "UD_Croatian-SET/hr_set-ud",
    "hye": "UD_Armenian-BSUT/hy_bsut-ud",
    #"kan": "UD_Kannada-MCG/kannada_mcg-ud",
    "nld": "UD_Dutch-LassySmall/nl_lassysmall-ud",
    "slk": "UD_Slovak-SNK/sk_snk-ud",
}


UD_TAGS = [
    "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
]


UD_TAG2ID = {tag: i for i, tag in enumerate(UD_TAGS)}


def read_conllu(file: IO) -> Tuple[List[List[str]], List[List[str]]]:
    logging.info("Reading CoNLL-U data from %s", file.name)
    sentences = []
    tag_sequences = []

    this_sentence = []
    this_tags = []
    for line in file:
        line = line.strip()
        if line:
            if line.startswith('#'):
                continue
            fields = line.split('\t')
            this_sentence.append(fields[1])
            this_tags.append(fields[3])
        else:
            if this_sentence:
                sentences.append(this_sentence)
                tag_sequences.append(this_tags)
            this_sentence = []
            this_tags = []

    if this_sentence:
        sentences.append(this_sentence)
        tag_sequences.append(this_tags)

    return sentences, tag_sequences


def subword_segment(
        sentences: List[List[str]],
        tag_sequences: List[List[str]],
        segment_fn: callable) -> Tuple[List[List[str]], List[List[str]], List[List[int]]]:
    segmented_sentences = []
    segmented_tags = []
    token_lens = []
    for sentence, tags in zip(sentences, tag_sequences):
        this_sentence = []
        this_tags = []
        this_token_lens = []
        for token, tag in zip(sentence, tags):
            subword_tokens = ["<sep>"] + segment_fn(token)
            this_sentence.extend(subword_tokens)
            this_tags.extend([tag] * len(subword_tokens))
            this_token_lens.append(len(subword_tokens))
        segmented_sentences.append(this_sentence)
        segmented_tags.append(this_tags)
        token_lens.append(this_token_lens)
    return segmented_sentences, segmented_tags, token_lens


def transformers_closure(tokenizer: PreTrainedTokenizerFast) -> callable:
    def transformers_segment(word):
        out_tokens = []
        try:
            for sbwrd in tokenizer.tokenize(word):
                if sbwrd.startswith("##"):
                    sbwrd = sbwrd[2:]
                elif sbwrd.startswith("▁"):
                    sbwrd = sbwrd[1:]
                if not sbwrd:
                    continue
                out_tokens.append(sbwrd)
            return out_tokens
        except Exception as e:
            return ["<unk>"]  # Fallback to <unk> if tokenization fails

    return transformers_segment


def identity_closure(word: str) -> List[str]:
    return [word]


def charachter_closure(word: str) -> List[str]:
    return list(word)


def legros_closure(
        subwords_file: str,
        subword_embeddings_file: str,
        fasttext_model_file: str) -> callable:
    """Create a Legros tokenizer segmentation function."""
    
    # Load vocabulary
    with open(subwords_file, 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f]
    vocab = Vocab(vocab_list)
    
    # Load subword embeddings (text format with space-separated values)
    subword_embeddings = []
    with open(subword_embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            embedding = [float(x) for x in line.strip().split()]
            subword_embeddings.append(embedding)
    subword_embeddings = np.array(subword_embeddings)
    
    # Load FastText model
    fasttext_model = FastText.load(fasttext_model_file)
    
    # Create subword to index mapping
    subwrd2idx = {subword: idx for idx, subword in enumerate(vocab_list)}
    
    def legros_segment(word):
        """Segment a word using Legros tokenizer."""
        try:
            from legros.segment_vocab_with_subword_embeddings import try_segment
            
            # Get word vector from FastText model
            try:
                word_vector = fasttext_model.wv[word]
            except KeyError:
                # If word not in vocabulary, use mean of all vectors as fallback
                word_vector = fasttext_model.wv.vectors.mean(0)
            
            # Get segmentation using Legros
            segmentations = try_segment(
                word,
                word_vector,  # Pass the vector directly instead of the model
                subwrd2idx,
                subword_embeddings,
                inference_mode="sum",
                sample=False,
                is_bert_wordpiece=False,
                exclude_subwords=None
            )
            
            if segmentations:
                # Get the best segmentation (first one)
                segmentation_str, _ = segmentations[0]
                segments = segmentation_str.split()
                return segments
            else:
                # Fallback to character-level segmentation
                return list(word)
                
        except Exception as e:
            logging.warning(f"Legros segmentation failed for word '{word}': {e}")
            return list(word)  # Fallback to character-level
    
    return legros_segment


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, tags = zip(*batch)
    lens = torch.tensor([len(token) for token in tokens])
    mask = torch.arange(max(lens)) < lens[:, None]
    tokens = torch.stack([torch.nn.functional.pad(token, (0, max(lens) - len(token))) for token in tokens])
    tags = torch.stack([torch.nn.functional.pad(tag, (0, max(lens) - len(tag))) for tag in tags])
    return tokens, mask, tags


def prepare_data(
        sentences: List[List[str]],
        tag_sequences: List[List[str]],
        vocab2id: Dict[str, int],
        batch_size: int,
        shuffle: bool=True) -> torch.utils.data.DataLoader:
    # Turn loaded data into PyTorch dataset
    data = []
    for sentence, tags in zip(sentences, tag_sequences):
        token_ids = torch.tensor([vocab2id.get(token, 1) for token in sentence])
        tag_ids = torch.tensor([UD_TAG2ID[tag] for tag in tags])
        data.append((token_ids, tag_ids))

    # Batch the dataset and turn it into a DataLoader
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn)
    return loader


def get_vocab(sentences: List[List[str]], max_size: int=None) -> Tuple[List[str], Dict[str, int]]:
    counter = defaultdict(int)
    for sentence in sentences:
        for token in sentence:
            counter[token] += 1
    vocab = ["<pad>", "<unk>", "<sep>"] + sorted(
        counter, key=counter.get, reverse=True)[:max_size]
    vocab2id = {token: i for i, token in enumerate(vocab)}
    return vocab, vocab2id


class POSTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Dropout(dropout))
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.hidden2tag = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tagset_size))

    def forward(self, sentence, mask):
        embeds = self.embedding(sentence)
        packed_embeds = pack_padded_sequence(embeds, mask.sum(1).tolist(), batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        tag_logits = self.hidden2tag(lstm_out)
        return tag_logits


def train_pos_tagger(
        lang: str,
        tokenizer_type: str,
        subword_model: str=None,
        legros_subwords: str=None,
        legros_embeddings: str=None,
        legros_fasttext: str=None,
        max_steps: int=3200,
        embedding_dim: int=300,
        hidden_dim: int=600,
        dropout: float=0.5,
        batch_size: int=256,
        lr: float=0.01,
        log_interval: int=10,
        eval_interval: int=200,
        word_vocab_size: int=32000,
        seed: int=42) -> None:

    # Ensure fixed seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(f"{UD_HOME}/{CODE_TO_NAME[lang]}-train.conllu") as f:
        train_sentences, train_tag_sequences = read_conllu(f)
    with open(f"{UD_HOME}/{CODE_TO_NAME[lang]}-dev.conllu") as f:
        dev_sentences, dev_tag_sequences = read_conllu(f)
    with open(f"{UD_HOME}/{CODE_TO_NAME[lang]}-test.conllu") as f:
        test_sentences, test_tag_sequences = read_conllu(f)
    orig_test_tag_sequences = test_tag_sequences
    test_token_lens = [[1 for _ in sent] for sent in test_sentences]

    if tokenizer_type == "words":
        segment_fn = identity_closure
    elif tokenizer_type == "char":
        segment_fn = charachter_closure
    elif tokenizer_type == "subwords":
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=subword_model,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
        )
        segment_fn = transformers_closure(tokenizer)
    elif tokenizer_type == "legros":
        if not all([legros_subwords, legros_embeddings, legros_fasttext]):
            raise ValueError("Legros tokenizer requires subwords, embeddings, and fasttext files")
        segment_fn = legros_closure(legros_subwords, legros_embeddings, legros_fasttext)
    else:
        raise ValueError(f"Unknown subword method: {tokenizer_type}")

    logging.info("Segmenting train sentences.")
    train_sentences, train_tag_sequences, _ = subword_segment(
        train_sentences, train_tag_sequences, segment_fn)
    logging.info("Segmenting dev sentences.")
    dev_sentences, dev_tag_sequences, _ = subword_segment(
        dev_sentences, dev_tag_sequences, segment_fn)
    logging.info("Segmenting test sentences.")
    test_sentences, test_tag_sequences, test_token_lens = subword_segment(
        test_sentences, test_tag_sequences, segment_fn)

    logging.info("Building vocabulary.")
    vocab, vocab2id = get_vocab(train_sentences, max_size=None if tokenizer_type == "words" else word_vocab_size)
    logging.info("Vocabulary size: %d", len(vocab))

    logging.info("Preparing pytorch data loaders.")
    train_loader = prepare_data(train_sentences, train_tag_sequences, vocab2id, batch_size)
    dev_loader = prepare_data(dev_sentences, dev_tag_sequences, vocab2id, batch_size)
    test_loader = prepare_data(test_sentences, test_tag_sequences, vocab2id, batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Building model, device: %s", device)
    model = POSTagger(len(vocab), len(UD_TAGS), embedding_dim, hidden_dim, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Random file name for saving the model
    model_identifier = subword_model or f"{tokenizer_type}_{legros_subwords or ''}"
    filename = f"{lang}_{tokenizer_type}_{hashlib.md5((model_identifier + str(seed)).encode('utf-8')).hexdigest()}.pt"

    logging.info("Training model.")
    step = 0
    best_loss = float('inf')
    for epoch in range(1000):
        model.train()
        for tokens, mask, tags in train_loader:
            tokens, mask, tags = tokens.to(device), mask.to(device), tags.to(device)
            optimizer.zero_grad()
            tag_logits = model(tokens, mask)
            loss = criterion(tag_logits.view(-1, len(UD_TAGS)), tags.view(-1))
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_interval == 0:
                logging.info("Epoch %d, step %d, loss %.4f", epoch, step, loss.item())

            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    total_correct = 0
                    total_words = 0
                    for tokens, mask, tags in dev_loader:
                        tokens, mask, tags = tokens.to(device), mask.to(device), tags.to(device)
                        tag_logits = model(tokens, mask)
                        loss = criterion(tag_logits.view(-1, len(UD_TAGS)), tags.view(-1))
                        total_loss += loss.item()

                        # Compute accuracy
                        _, predicted = tag_logits.max(2)
                        correct = (predicted == tags).float() * mask
                        total_correct += correct.sum().item()
                        total_words += mask.sum().item()

                    accuracy = total_correct / total_words
                    if total_loss < best_loss:
                        best_loss = total_loss
                        torch.save(model.state_dict(), filename)
                    logging.info("Validation loss %.4f, accuracy %.4f", total_loss / len(dev_loader), accuracy)
                model.train()

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    logging.info("Training done, loading best model.")
    model.load_state_dict(torch.load(filename))
    os.remove(filename)
    model.eval()

    logging.info("Evaluating model.")
    predicted_distributions = []
    for tokens, mask, _ in test_loader:
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            distributions = nn.functional.softmax(model(tokens, mask), dim=2).cpu().numpy()
        for sent_mask, sent_distr in zip(mask, distributions):
            sent_len = sent_mask.sum().item()
            predicted_distributions.append(sent_distr[:sent_len])

    total_tag_counts = 0
    correct_tag_counts = 0
    for gold_tags, lens, distributions in zip(
            orig_test_tag_sequences, test_token_lens, predicted_distributions):
        start_idx = 0
        assert len(gold_tags) == len(lens)
        assert sum(lens) == len(distributions)
        for tag, length in zip(gold_tags, lens):
            mean_distr = distributions[start_idx:start_idx + length].mean(0)
            predicted_tag = UD_TAGS[mean_distr.argmax()]
            if predicted_tag == tag:
                correct_tag_counts += 1
            total_tag_counts += 1
            start_idx += length

    logging.info("Accuracy: %.2f%%", 100 * correct_tag_counts / total_tag_counts)
    logging.info("Done.")
    return correct_tag_counts / total_tag_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('lang', help='language code')
    parser.add_argument('--tokenizer-type', type=str, default="words", help='subword tokenization', choices=["subwords", "char", "legros"])
    parser.add_argument('--subword-model', type=str, default=None, help='subword model')
    parser.add_argument('--legros-subwords', type=str, default=None, help='Legros subwords file')
    parser.add_argument('--legros-embeddings', type=str, default=None, help='Legros subword embeddings file')
    parser.add_argument('--legros-fasttext', type=str, default=None, help='Legros FastText model file')
    parser.add_argument('--max-steps', type=int, default=3200, help='max steps')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--embedding-dim', type=int, default=300, help='embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=600, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--log-interval', type=int, default=10, help='log interval')
    parser.add_argument('--eval-interval', type=int, default=200, help='evaluation interval')
    parser.add_argument('--word-vocab-size', type=int, default=32000, help='word vocab size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    train_pos_tagger(
        args.lang, args.tokenizer_type,
        subword_model=args.subword_model,
        legros_subwords=args.legros_subwords,
        legros_embeddings=args.legros_embeddings,
        legros_fasttext=args.legros_fasttext,
        max_steps=args.max_steps,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        word_vocab_size=args.word_vocab_size,
        seed=args.seed)
