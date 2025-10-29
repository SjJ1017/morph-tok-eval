import argparse
import pandas as pd
from collections import defaultdict
import re
import json
import os
import sys
import pandas as pd
import numpy as np


DATASETS = {
    "ces": "ces-unimorph2uniseg_derinet",
    "deu": "deu-unimorph2uniseg_CELEX",
    "eng": "eng-unimorph2uniseg_CELEX",
    "fin": "fin-unimorph2uniseg_morphynet",
    "hbs": "hbs-unimorph2uniseg_MorphyNet",
    "hye": "hye-unimorph2uniseg",
    "nld": "nld-unimorph2uniseg_CELEX",
    "slk": "slk-unimorph2olostiak",
}

TOKENIZERS = ["bpe", "unigram", "wordpiece"]
VOCAB_SIZES = [2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80]


def load_json_files_to_dataframe(file_list):
    # Dictionary to store data grouped by tokenizer
    tokenizer_data = defaultdict(dict)

    # Iterate through all JSON files in the directory
    for filename in file_list:
        # Load the JSON content
        with open(filename, 'r') as file:
            json_content = json.load(file)

        # String .json from the filename
        tokenizer_name = re.sub(r"^evaluated/.*/|\.json$", "", filename)

        # Throw away stuff starting with 'gold'
        json_content = {k: v for k, v in json_content.items() if not k.startswith('gold')}

        # Update the tokenizer's data
        tokenizer_data[tokenizer_name] = json_content

    # Convert the processed data to a DataFrame
    rows = {}
    for tokenizer_name, content in tokenizer_data.items():
        row = {'tokenizer': tokenizer_name}
        row.update(content)
        rows[tokenizer_name] = row

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='Language code')
    parser.add_argument('output', type=str, help='Output file name', default=sys.stdout, nargs='?')
    args = parser.parse_args()

    evaluation_files = [f"evaluated/{DATASETS[args.lang]}/{tokenizer}-{vocab_size}k.json"
                        for tokenizer in TOKENIZERS
                        for vocab_size in VOCAB_SIZES]

    rows = load_json_files_to_dataframe(evaluation_files)

    for tokenizer in TOKENIZERS:
        for vocab_size in VOCAB_SIZES:
            filname = f"pos_tagging/{args.lang}/{tokenizer}-{vocab_size}k.tsv"
            if not os.path.exists(filname):
                continue
            with open(filname, 'r') as file:
                json_content = json.load(file)
            rows[f"{tokenizer}-{vocab_size}k"]["pos_accuracy"] = json_content["avg_accuracy"]

    # Create DataFrame
    df = pd.DataFrame(rows.values())

    # Ensure tokenizer is the first column
    if not df.empty:
        columns = df.columns.tolist()
        columns.remove('tokenizer')
        df = df[['tokenizer'] + columns]

    test_columns = [
            "avg_segments", "boundary_precision",
            "boundary_recall", "boundary_f_score"] + [col for col in df.columns if col.startswith('test-')]
    other_columns = ["pos_accuracy"]

    correlation_df = pd.DataFrame(
        index=test_columns,
        columns=other_columns,
        dtype=float
    )

    # Calculate correlations
    for test_col in test_columns:
        for other_col in other_columns:
            correlations = []
            for vocab_size in VOCAB_SIZES:
                filtered_df = df[df['tokenizer'].str.contains(f"-{vocab_size}k")]
                correlation = filtered_df[test_col].corr(filtered_df[other_col], method='kendall')
                correlations.append(correlation)
            correlation_df.at[test_col, other_col] = np.mean(correlations)
    correlation_df.to_csv(args.output, sep='\t', index=True, header=True)

if __name__ == "__main__":
    main()

