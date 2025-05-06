LANGUAGES = ["ces", "fin", "hye", "kan"]

LNG_CODES = {
    "ces": "cs", # Czech
    "fin": "fi", # Finnish
    "hye": "hy", # Armenian
    "kan": "kn", # Kannada
}


VOCAB_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96]


rule all:
    input:
        expand("tokenizers/{lng}/{tok_type}-{vocab_size}k/vocab.json",
        lng=LANGUAGES, vocab_size=VOCAB_SIZES, tok_type=["bpe", "unigram"])


rule download_cc100:
    output:
        "data/cc100/{lng}.txt"
    params:
        lng = lambda wildcards: LNG_CODES[wildcards.lng],
    shell:
        """
        mkdir -p data/cc100
        wget https://data.statmt.org/cc-100/{params.lng}.txt.xz -O data/cc100/{wildcards.lng}.tmp.xz
        unxz data/cc100/{wildcards.lng}.tmp.xz
        head -n 1M data/cc100/{wildcards.lng}.tmp > {output}
        rm data/cc100/{wildcards.lng}.tmp
        """


def train_tokenizer(input_file, save_path, vocab_size, model_type):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, Unigram
    from tokenizers.normalizers import NFD
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer, UnigramTrainer

    model_class = BPE if model_type == "bpe" else Unigram
    trainer_class = BpeTrainer if model_type == "bpe" else UnigramTrainer

    tokenizer = Tokenizer(model_class())
    
    # Customize the tokenizer
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = Whitespace()
    
    # Initialize the trainer
    trainer = trainer_class(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<mask>"]
    )
    
    # Train the tokenizer
    tokenizer.train(files=[input_file], trainer=trainer)

    # Save the tokenizer
    tokenizer.save(save_path)
    print(f"{model_type} tokenizer with vocab size {vocab_size} saved to {save_path}")


rule train_bpe:
    input:
        "data/cc100/{lng}.txt"
    output:
        "tokenizers/{lng}/bpe-{vocab_size}k/vocab.json"
    run:
        train_tokenizer(
            input_file=input[0],
            save_path=output[0],
            vocab_size=1000 * int(wildcards.vocab_size),
            model_type="bpe"
        )


rule train_unigram:
    input:
        "data/cc100/{lng}.txt"
    output:
        "tokenizers/{lng}/unigram-{vocab_size}k/vocab.json"
    run:
        train_tokenizer(
            input_file=input[0],
            save_path=output[0],
            vocab_size=1000 * int(wildcards.vocab_size),
            model_type="unigram"
        )