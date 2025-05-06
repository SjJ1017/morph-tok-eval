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


rule train_tokenizer:
    input:
        "data/cc100/{lng}.txt"
    output:
        "tokenizers/{lng}/{tokenizer_type}-{vocab_size}k.json"
    wildcard_constraints:
        tokenizer_type="bpe|unigram"
    run:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, Unigram
    from tokenizers.normalizers import NFD
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer, UnigramTrainer

        model_class = BPE if wildcards.tokenizer_type == "bpe" else Unigram
        trainer_class = BpeTrainer if wildcards.tokenizer_type == "bpe" else UnigramTrainer

    tokenizer = Tokenizer(model_class())
    
    # Customize the tokenizer
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = Whitespace()
    
    # Initialize the trainer
    trainer = trainer_class(
            vocab_size=1000 * int(wildcards.vocab_size),
        min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<mask>"]
    )
    
    # Train the tokenizer
        tokenizer.train(files=input, trainer=trainer)

    # Save the tokenizer
        tokenizer.save(output[0])
        print(f"{wildcards.tokenizer_type} tokenizer with vocab size {wildcards.vocab_size}k saved to {output[0]}")