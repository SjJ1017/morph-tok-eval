LANGUAGES = ["ces", "fin", "hye", "kan"]

LNG_CODES = {
    "ces": "cs", # Czech
    "fin": "fi", # Finnish
    "hye": "hy", # Armenian
    "kan": "kn", # Kannada
}


VOCAB_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]


PRE_TRAINED_TOKENIZERS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "xlmv": "facebook/xlm-v-base",
    "bloom": "bigscience/bloom",
    "xglm": "facebook/xglm-564M",
    "mt5": "google/mt5-base",
    "mbart": "facebook/mbart-large-50",
    "m2m": "facebook/m2m100_418M",
    "nllb": "facebook/nllb-200-1.3B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama2": "meta-llama/Llama-2-7b",
    "llama3": "meta-llama/Llama-3-7b",
    "gemma3": "google/gemma-3-1b-it",
    "qwen3": "Qwen/Qwen3-0.6B",
}


localrules: tokenize_unimorph_our_tokenizer, tokenize_unimorph_huggingface


rule all:
    input:
        expand("segmented/{lng}/{tok_type}-{vocab_size}k.tsv",
            lng=LANGUAGES, vocab_size=VOCAB_SIZES, tok_type=["bpe", "unigram"]),
        expand("segmented/{lng}/pretrained-{tokenizer}.tsv",
            lng=LANGUAGES, tokenizer=PRE_TRAINED_TOKENIZERS.keys()),


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
    resources:
        mem="16G",
        tasks=1,
        cpus_per_task=2,
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


def tokenize_unimorph(input_file, output_file, tokenizer):
    with open(input_file, encoding='UTF-8') as f_in, open(output_file, 'w', encoding='UTF-8') as f_out:
        for line in f_in:
            word, tag, segments = line.split("\t")
            segments = '|'.join(tokenizer.tokenize(word))  
            print(word, tag, segments, sep='\t', file=f_out)


rule tokenize_unimorph_our_tokenizer:
    input:
        data="data/{lng}/{lng}.tsv",
        tokenizer="tokenizers/{lng}/{tokenizer_type}-{vocab_size}k.json"
    output:
        "segmented/{lng}/{tokenizer_type}-{vocab_size}k.tsv"
    run:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=input.tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
        )

        tokenize_unimorph(
            input_file=input.data,
            output_file=output[0],
            tokenizer=tokenizer
        )


rule tokenize_unimorph_huggingface:
    input:
        data="data/{lng}/{lng}.tsv"
    output:
        "segmented/{lng}/pretrained-{tokenizer}.tsv"
    wildcard_constraints:
        tokenizer="|".join(PRE_TRAINED_TOKENIZERS.keys())
    run:
        from transformers import AutoTokenizer
        full_tokenizer_id = PRE_TRAINED_TOKENIZERS[wildcards.tokenizer]
        tokenizer = AutoTokenizer.from_pretrained(full_tokenizer_id)

        tokenize_unimorph(
            input_file=input.data,
            output_file=output[0],
            tokenizer=tokenizer
        )