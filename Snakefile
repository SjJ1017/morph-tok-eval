
LANGUAGES = ["ces", "fin", "hye", "kan"]

LNG_CODES = {
    "ces": "cs", # Czech
    "fin": "fi", # Finnish
    "hye": "hy", # Armenian
    "kan": "kn", # Kannada
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


THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


localrules: tokenize_unimorph_our_tokenizer, tokenize_unimorph_huggingface


rule all:
    input:
        expand("correlations/{lng}.txt", lng=LANGUAGES),


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


def unicode_safe_tokenize(tokenizer, text):
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


rule evaluate_segmentation:
    input:
        gold_data="data/{lng}/{lng}.tsv",
        segmented_data="segmented/{lng}/{segmented_file}.tsv"
    output:
        "evaluated/{lng}/{segmented_file}-{threshold}.json"
    run:
        import json
        from align import evaluate_segmentations
        results = evaluate_segmentations(
            10, float(wildcards.threshold), input.gold_data, input.segmented_data)
        with open(output[0], 'w', encoding='UTF-8') as f_out:
            json.dump(results, f_out, ensure_ascii=False, indent=4)


def load_json_files_to_dataframe(file_list):
    from collections import defaultdict
    import re
    import json
    import pandas as pd

    # Dictionary to store data grouped by tokenizer
    tokenizer_data = defaultdict(dict)
    
    # Pattern to extract tokenizer and threshold from filename
    pattern = r"(.*)-(\d+\.\d+)\.json"
    
    # Iterate through all JSON files in the directory
    for filename in file_list:
        # Extract tokenizer and threshold from filename
        match = re.match(pattern, filename)
        if match:
            tokenizer_name = match.group(1)
            threshold = float(match.group(2))
            
            # Load the JSON content
            with open(filename, 'r') as file:
                json_content = json.load(file)
            
            # Process the JSON content according to requirements
            processed_content = {}
            
            for key, value in json_content.items():
                if key.startswith('gold'):
                    # Skip items starting with 'gold'
                    continue
                elif key.startswith('test'):
                    # For 'test' items, append the threshold
                    new_key = f"{key}_{threshold}"
                    processed_content[new_key] = value
                else:
                    # For other items, keep them as is
                    processed_content[key] = value
            processed_content['threshold'] = threshold
            
            # Update the tokenizer's data
            tokenizer_data[tokenizer_name].update(processed_content)
    
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
        our_tokenizers=expand("evaluated/{{lng}}/{tok_type}-{vocab_size}k-{threshold}.json",
            vocab_size=VOCAB_SIZES, tok_type=["bpe", "unigram"], threshold=THRESHOLDS),
        pretrained_tokenizers=expand("evaluated/{{lng}}/pretrained-{tokenizer}-{threshold}.json",
            tokenizer=PRE_TRAINED_TOKENIZERS.keys(), threshold=THRESHOLDS),
    output:
        "correlations/{lng}.txt"
    run:
        import pandas as pd

        df = load_json_files_to_dataframe(input.our_tokenizers) #+ input.pretrained_tokenizers)

        # Identify columns that start with 'test-'
        test_columns = ["avg_segments"] + [col for col in df.columns if col.startswith('test-')]
        
        # Identify other columns (excluding 'tokenizer' which is likely non-numeric)
        other_columns = ["boundary_precision", "boundary_recall", "boundary_f_score"]
        
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