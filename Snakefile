LANGUAGES = ["ces", "fin", "hye", "kan", "deu", "eng", "hbs", "nld", "slk"]

LNG_CODES = {
    "ces": "cs", # Czech
    "fin": "fi", # Finnish
    "hye": "hy", # Armenian
    "kan": "kn", # Kannada
	"deu": "de", # German
    "eng": "en", # English
    "hbs": "hr", # Serbo-Croatian
    "nld": "nl", # Dutch
    "slk": "sk", # Slovak
}


VOCAB_SIZES = [2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]


PRE_TRAINED_TOKENIZERS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "xlmv": "facebook/xlm-v-base",
    "bloom": "bigscience/bloom",
    "xglm": "facebook/xglm-564M",
    "mt5": "google/mt5-base",
    "mbart": "facebook/mbart-large-50",
    #"m2m": "facebook/m2m100_418M",  # Does no support offset mapping
    "nllb": "facebook/nllb-200-1.3B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "llama2": "meta-llama/Llama-2-7b",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2": "google/gemma-2-2b-it",
    "gemma3": "google/gemma-3-1b-it",
    "qwen2": "Qwen/Qwen2.5-Omni-3B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "falcon": "tiiuae/falcon-7b-instruct",
}

MULTIBLIMP_SCORES = {
    "ces": 97.2, #EuroLLM 9B
    "deu": 99.0, #Llama 3 70B
    "eng": 99.5, #OLMo2 32B
    "fin": 96.4, #Gemma3 27B
    "hbs": 98.3, #Gemma3 27B
    "hye": 98.4, #Llama 3 70B
    "kan": 00.0, # No score available
    "nld": 100.0, #Llama 3 8B
    "slk": 96.3, # EuroLLM 9B
}

DATASETS = [
	"ces-unimorph2uniseg_derinet",
	"ces-unimorph",
	"deu-unimorph2uniseg_CELEX",
	"eng-unimorph2uniseg_CELEX",
	"fin-unimorph2uniseg_morphynet",
	"fin-unimorph",
	"hbs-unimorph2uniseg_MorphyNet",
	"hye-unimorph2uniseg",
	"kan-unimorph2uniseg_KCIS",
	"nld-unimorph2uniseg_CELEX",
    "slk-unimorph2olostiak",
]


THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


IBM_MODELS = ["IBM1", "IBM2"]


localrules: tokenize_unimorph_our_tokenizer, tokenize_unimorph_huggingface, character_tokenization, gold_tokenization, compute_correlations


rule all:
    input:
        expand("correlations/{dataset}.txt", dataset=DATASETS),
        #expand("pos_tagging/{lng}/{tokenizer_prefix}{tokenizer_type}-{vocab_size}k.tsv",
        #    lng=[l for l in LANGUAGES if l != "kan"],
        #    tokenizer_type=["bpe", "unigram", "wordpiece"],
        #    tokenizer_prefix=["", "legros-"],
        #    vocab_size=VOCAB_SIZES),


rule download_cc100:
    output:
        "data/cc100/{lng}.txt"
    params:
        lng = lambda wildcards: LNG_CODES[wildcards.lng],
    resources:
        mem="16G",
        tasks=1,
        cpus_per_task=8,
    shell:
        """
        mkdir -p data/cc100
        wget https://data.statmt.org/cc-100/{params.lng}.txt.xz -O data/cc100/{wildcards.lng}.tmp.xz
        xzcat data/cc100/{wildcards.lng}.tmp.xz | head -n 1M > data/cc100/{wildcards.lng}.txt || true
        rm data/cc100/{wildcards.lng}.tmp.xz
        """


rule train_tokenizer:
    input:
        "data/cc100/{lng}.txt"
    output:
        "tokenizers/{lng}/{tokenizer_type}-{vocab_size}k.json"
    wildcard_constraints:
        tokenizer_type="bpe|unigram|wordpiece",
    resources:
        mem="16G",
        tasks=1,
        cpus_per_task=2,
    run:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE, Unigram, WordPiece
        from tokenizers.normalizers import NFD
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

        if wildcards.tokenizer_type == "bpe":
            model_class = BPE
            trainer_class = BpeTrainer
        elif wildcards.tokenizer_type == "unigram":
            model_class = Unigram
            trainer_class = UnigramTrainer
        elif wildcards.tokenizer_type == "wordpiece":
            model_class = WordPiece
            trainer_class = WordPieceTrainer
        else:
            raise ValueError(f"Unknown tokenizer type: {wildcards.tokenizer_type}")

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


def unicode_safe_tokenize(tokenizer, text):
    try:
        # Get the encoding with offsets
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)

        offset_mapping = encoding["offset_mapping"]
        # Fix the offset mapping, so it is non-overlapping and covers the whole string
        for i in range(len(offset_mapping) - 1):
            this_start, this_end = offset_mapping[i]
            next_start, next_end = offset_mapping[i + 1]
            if this_end != next_start:
                offset_mapping[i + 1] = (this_end, next_end)

        # Extract tokens directly from the original text using the offsets
        original_tokens = [text[start:end] for start, end in offset_mapping]
        return original_tokens #, standard_tokens
    except:
        # If the tokenizer fails, fall back to character-level tokenization
        return list(text)  # Return characters as tokens


def tokenize_unimorph(input_file, output_file, tokenizer):
    with open(input_file, encoding='UTF-8') as f_in, open(output_file, 'w', encoding='UTF-8') as f_out:
        for line in f_in:
            word, tag, segments = line.split("\t")
            tokenized = unicode_safe_tokenize(tokenizer, word)
            assert "".join(tokenized) == word, f"Tokenization mismatch: {word} != {' '.join(tokenized)}"
            segments = '|'.join(tokenized)

            print(word, tag, segments, sep='\t', file=f_out)


rule tokenize_unimorph_our_tokenizer:
    input:
        data="data/morpho/{lng}-{dataset_type}.tsv",
        tokenizer="tokenizers/{lng}/{tokenizer_type}-{vocab_size}k.json"
    output:
        "segmented/{lng}-{dataset_type}/{tokenizer_type}-{vocab_size}k.tsv"
    wildcard_constraints:
        lng="|".join(LNG_CODES.keys()),
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
        data="data/morpho/{lng}-{dataset_type}.tsv"
    output:
        "segmented/{lng}-{dataset_type}/pretrained-{tokenizer}.tsv"
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


rule character_tokenization:
    input:
        data="data/morpho/{dataset}.tsv"
    output:
        "segmented/{dataset}/char.tsv"
    run:
        with open(input.data, encoding='UTF-8') as f_in, open(output[0], 'w', encoding='UTF-8') as f_out:
            for line in f_in:
                word, tag, segments = line.split("\t")
                segments = '|'.join(list(word))
                print(word, tag, segments, sep='\t', file=f_out)


rule gold_tokenization:
    input:
        data="data/morpho/{dataset}.tsv"
    output:
        "segmented/{dataset}/gold.tsv"
    shell:
        """
        mkdir -p segmented/{wildcards.dataset}
        cp {input.data} {output}
        """

rule evaluate_segmentation:
    input:
        gold_data="data/morpho/{dataset}.tsv",
        segmented_data="segmented/{dataset}/{segmented_file}.tsv"
    output:
        "evaluated/{dataset}/{segmented_file}.json"
    resources:
        mem="4G",
        tasks=1,
        cpus_per_task=2,
    run:
        import json
        from align import evaluate_segmentations
        results = {}
        for model in IBM_MODELS:
            results.update(evaluate_segmentations(
                input.gold_data, input.segmented_data, THRESHOLDS, 10,
                model, skip_gold_train=True))
        with open(output[0], 'w', encoding='UTF-8') as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)


def load_json_files_to_dataframe(file_list):
    from collections import defaultdict
    import re
    import json
    import pandas as pd

    # Dictionary to store data grouped by tokenizer
    tokenizer_data = defaultdict(dict)

    # Iterate through all JSON files in the directory
    for filename in file_list:
        # Load the JSON content
        with open(filename, 'r') as file:
            json_content = json.load(file)

        # String .json from the filename
        tokenizer_name = re.sub(r"^evaluated/|\.json$", "", filename)

        # Throw away stuff starting with 'gold'
        json_content = {k: v for k, v in json_content.items() if not k.startswith('gold')}

        # Update the tokenizer's data
        tokenizer_data[tokenizer_name] = json_content

    # Convert the processed data to a DataFrame
    rows = []
    for tokenizer_name, content in tokenizer_data.items():
        row = {'tokenizer': tokenizer_name}
        row.update(content)
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Ensure tokenizer is the first column
    if not df.empty:
        columns = df.columns.tolist()
        columns.remove('tokenizer')
        df = df[['tokenizer'] + columns]

    return df


rule compute_correlations:
    input:
        gold_tokenizer="evaluated/{dataset}/gold.json",
        char_tokenizer="evaluated/{dataset}/char.json",
        our_tokenizers=expand("evaluated/{{dataset}}/{prefix}{tok_type}-{vocab_size}k.json",
            vocab_size=VOCAB_SIZES, tok_type=["bpe", "unigram", "wordpiece"], prefix=["", "legros-"]),
        #pretrained_tokenizers=expand("evaluated/{{dataset}}/pretrained-{tokenizer}.json",
        #    tokenizer=PRE_TRAINED_TOKENIZERS.keys()),
    output:
        "correlations/{dataset}.txt"
    run:
        import pandas as pd

        df = load_json_files_to_dataframe(
            [input.gold_tokenizer, input.char_tokenizer] +
            input.our_tokenizers) #+ input.pretrained_tokenizers)

        lang_code = wildcards.dataset[:3]
        
        if lang_code in LANGUAGE_SCORES:
            df['language_score'] = LANGUAGE_SCORES[lang_code]
        else:
            df['language_score'] = None

        # Identify columns that start with 'test-'
        test_columns = ["avg_segments", "segments_ratio"] + [col for col in df.columns if col.startswith('test-')]

        # Identify other columns (excluding 'tokenizer' which is likely non-numeric)
        other_columns = ["boundary_precision", "boundary_recall", "boundary_f_score", "language_score"]

        # Create an empty DataFrame to store correlations
        correlation_df = pd.DataFrame(
            index=test_columns,
            columns=other_columns,
            dtype=float
        )

        # Calculate correlations
        for test_col in test_columns:
            for other_col in other_columns:
                # Computer Spearman correlation
                correlation = df[test_col].corr(df[other_col], method='spearman')
                correlation_df.at[test_col, other_col] = correlation

        # Save the correlation DataFrame to a text file
        correlation_df.to_csv(output[0], sep='\t', index=True, header=True)
        print(f"Correlations saved to {output[0]}")


rule train_pos_tagger:
    input:
        tokenizer="tokenizers/{lng}/{tokenizer_type}-{vocab_size}k.json",
    output:
        "pos_tagging/{lng}/{tokenizer_type}-{vocab_size}k.tsv"
    resources:
        mem="16G",
        tasks=1,
        cpus_per_task=4,
        slurm_partition="gpu-ms,gpu-troja",
        constraint="gpuram16G|gpuram24G",
        slurm_extra="--gres=gpu:1"
    run:
        from pos_tagger import train_pos_tagger
        import json

        accuracies = []
        for i in range(5):
            accuracy = train_pos_tagger(
                wildcards.lng, tokenizer_type="subwords",
                subword_model=input.tokenizer,
                seed=1348 + i)
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)
        with open(output[0], 'w', encoding='UTF-8') as f_out:
            json.dump({
                    "accuracies": accuracies,
                    "avg_accuracy": avg_accuracy},
                f_out, ensure_ascii=False, indent=4)


rule train_pos_tagger_legros:
    input:
        subwords="tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/subwords.19",
        embeddings="tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/subword_embeddings.19",
        fasttext="data/fasttext/{lng}/fasttext",
    output:
        "pos_tagging/{lng}/legros-{tokenizer_type}-{vocab_size}k.tsv"
    resources:
        mem="16G",
        tasks=1,
        cpus_per_task=4,
        slurm_partition="gpu-ms,gpu-troja",
        constraint="gpuram16G|gpuram24G",
        slurm_extra="--gres=gpu:1"
    run:
        from pos_tagger import train_pos_tagger
        import json

        accuracies = []
        for i in range(5):
            accuracy = train_pos_tagger(
                wildcards.lng, tokenizer_type="legros",
                legros_subwords=input.subwords,
                legros_embeddings=input.embeddings,
                legros_fasttext=input.fasttext,
                seed=1348 + i)
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)
        with open(output[0], 'w', encoding='UTF-8') as f_out:
            json.dump({
                    "accuracies": accuracies,
                    "avg_accuracy": avg_accuracy},
                f_out, ensure_ascii=False, indent=4)


rule pretokenize_corpus_for_fasttext:
    input: "data/cc100/{lng}.txt"
    output: "data/cc100/{lng}.txt.tok"
    threads: 8
    resources:
        mem="32G",
        cpus_per_task=8
    shell:
        "cat {input} | sacremoses -j {threads} -l {wildcards.lng} tokenize -x > {output}"

legros_home = "legros"  # Adjust this path to your Legros installation directory

rule train_fasttext:
    input: "data/cc100/{lng}.txt.tok"
    output:
        model="data/fasttext/{lng}/fasttext",
        aux="data/fasttext/{lng}/fasttext.syn1neg.npy",
        vocab="data/fasttext/{lng}/fasttext.vocab",
        pinv="data/fasttext/{lng}/fasttext.out_inv.txt",
        text="data/fasttext/{lng}/fasttext.txt",
        wvvocab="data/fasttext/{lng}/fasttext.wv.vectors_vocab.npy",
        wvngrams="data/fasttext/{lng}/fasttext.wv.vectors_ngrams.npy"
    threads: 16
    resources:
        mem="32G",
        cpus_per_task=16
    params:
        dimension=200,
        vocab_size=200000,
        epochs=10
    shell:
        """
        python3 {legros_home}/scripts/train_fasttext.py {input} {output.model} \
            --num-threads {threads} \
            --dimension {params.dimension} \
            --vocab-size {params.vocab_size} \
            --epochs {params.epochs}
        """


rule compile_legros:
    output:
        "{legros_home}/build/legros-train"
    resources:
        mem="16G",
        tasks=1,
        cpus_per_task=4,
    shell:
        """
        if [ ! -d {legros_home} ]; then
            git submodule update --init --recursive
        fi

        mkdir -p {legros_home}/build
        cd {legros_home}/build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j 4
        """


rule allowed_init_from_tokenizer:
    input:
        vocab="data/fasttext/{lng}/fasttext.vocab",
        tokenizer="tokenizers/{lng}/{tokenizer_type}-{size_k}k.json"
    output:
        "tokenizers/{lng}/legros-{tokenizer_type}-{size_k}k/init.allowed"
    params:
        expdir=lambda wildcards, output: output[0][:-13]
    threads: 2
    resources:
        mem="10G",
        cpus_per_task=2
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

        with open(input.vocab, 'r', encoding='UTF-8') as f_vocab, \
             open(output[0], 'w', encoding='UTF-8') as f_out:
            for line in f_vocab:
                word = line.strip()
                tokenized = unicode_safe_tokenize(tokenizer, word)
                print(' '.join(tokenized), file=f_out)


rule train_legros:
    input:
        legros_binary=legros_home + "/build/legros-train",
        data="data/cc100/{lng}.txt",
        ft_emb_text="data/fasttext/{lng}/fasttext.txt",
        ft_pinv="data/fasttext/{lng}/fasttext.out_inv.txt",
        allowed="tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/init.allowed"
    output:
        "tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/segmentations.19",
        "tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/subwords.19",
        "tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/unigram_stats.19",
        "tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/bigram_stats.19",
        "tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/subword_embeddings.19",
    threads: 60
    resources:
        mem="250G",
        cpus_per_task=60
    params:
        epochs=20,
        output_dir=lambda wildcards, output: output[0][:-17]
    shell:
        """
        {input.legros_binary} \
            {input.ft_emb_text} \
            {input.data} \
            --fastext-output-pseudoinverse {input.ft_pinv} \
            --allowed-substrings {input.allowed} \
            --epochs {params.epochs} \
            --output-directory {params.output_dir}

        rm {params.output_dir}/{{segmentations,subwords,unigram_stats,bigram_stats,subword_embeddings}}.{{1..18}}
        """


rule tokenize_with_legros_tokenizer:
    input:
        data="data/morpho/{lng}-{dataset_type}.tsv",
        subwords="tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/subwords.19",
        subword_embeddings="tokenizers/{lng}/legros-{tokenizer_type}-{vocab_size}k/subword_embeddings.19",
        fasttext="data/fasttext/{lng}/fasttext",
    output:
        "segmented/{lng}-{dataset_type}/legros-{tokenizer_type}-{vocab_size}k.tsv"
    wildcard_constraints:
        lng="|".join(LNG_CODES.keys()),
    threads: 1
    resources:
        mem="10G",
        cpus_per_task=1
    shell:
        """
        cut -f1 {input.data} | \
            PYTHONPATH={legros_home}/python python -m legros.segment_vocab_with_subword_embeddings \
                {input.fasttext} \
                {input.subwords} \
                {input.subword_embeddings} | \
            sed 's/ /|/g' | \
            paste  <(cut -f1,2 {input.data}) - > {output}
        """
