from collections import defaultdict
import argparse
import json
import logging
import re
import numpy as np


# Logging with date and time
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class IBM1:
    def __init__(self, num_iterations=10, split_tags=False):
        self.translation_probs = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.num_iterations = num_iterations
        self.split_tags = split_tags

    def train(self, data):
        for _ in range(self.num_iterations):
            count_st = defaultdict(lambda: defaultdict(float))
            total_s = defaultdict(float)

            for _, full_tags, split_tags, segments in data:
                if self.split_tags:
                    tags = split_tags
                else:
                    tags = full_tags
                for tag in tags:  
                    norm_factor = sum(self.translation_probs[s][tag] for s in segments) + 1e-10  

                    for s in segments:  
                        delta = self.translation_probs[s][tag] / norm_factor
                        count_st[s][tag] += delta
                        total_s[tag] += delta

            for s in count_st:
                for tag in count_st[s]:
                    self.translation_probs[s][tag] = count_st[s][tag] / (total_s[tag] + 1e-10)  
   
    def get_prob(self, segment, tags):
        if isinstance(tags, list): 
            return {tag: self.translation_probs[segment][tag] for tag in tags}
        else:  
            return self.translation_probs[segment][tags]

def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            word, tag, segmentation = line.split("\t")
            full_tags = tag.split("|")
            split_tags = re.split(r'\||;', tag)
            segments = segmentation.strip().split("|")  
            data.append((word, full_tags, split_tags, segments))      
    
    return data


def geometric_mean(numbers):
    return np.exp(np.log(numbers).mean())


def harmonic_mean(numbers):
    return len(numbers) / np.sum(1.0 / np.array(numbers))


def compute_score(data, em_segmenter, threshold):
    results = defaultdict(float)
    total_segments = 0
    for word, full_tags, split_tags, segmentation in data:
        for segment in segmentation:
            total_segments += 1
            if em_segmenter.split_tags:
                tags = split_tags
                suffix = "split"
            else:
                tags = full_tags
                suffix = "full"
            prob = em_segmenter.get_prob(segment, tags)
            scores = [x for x in prob.values() if x > threshold]

            if len(scores) > 0:
                results[f'morpho-score-sum-{suffix}'] += sum(scores)
                results[f'morpho-score-logsum-{suffix}'] += sum(np.log(scores))
                results[f'morpho-score-mean-{suffix}'] += sum(scores) / len(scores)
                results[f'morpho-score-harmonic-{suffix}'] += harmonic_mean(scores)
                results[f'morpho-score-geometric-{suffix}'] += geometric_mean(scores)
                results[f'morpho-score-min-{suffix}'] += min(scores)
                results[f'morpho-score-max-{suffix}'] += max(scores)

    for k in results:
        results[k] /= total_segments
    
    return results


def boundary_positions(word_segments):
    boundaries = set()
    idx = 0
    for segment in word_segments:  
        idx += len(segment)
        boundaries.add(idx)
    return boundaries


def evaluate_segmentations(gold_file, test_file, thresholds, iterations, skip_gold_train=False):
    results = {}

    gold_data = read_data(gold_file)
    if not skip_gold_train:
        logging.info("Training IBM Model 1 on gold data with full tags")
        em_segmenter_full = IBM1(num_iterations=iterations)
        em_segmenter_full.train(gold_data)
        logging.info("Computing scores for gold data with full tags")
        for threshold in thresholds:
            gold_scores_full = compute_score(gold_data, em_segmenter_full, threshold)
            for k, val in gold_scores_full.items():
                results[f"gold-{k}-{threshold}"] = val

        logging.info("Training IBM Model 1 on gold data with split tags")
        em_segmenter_split = IBM1(num_iterations=iterations, split_tags=True)
        em_segmenter_split.train(gold_data)
        logging.info("Computing scores for gold data with split tags")
        for threshold in thresholds:
            gold_scores_split = compute_score(gold_data, em_segmenter_split, threshold)
            for k, val in gold_scores_split.items():
                results[f"gold-{k}-{threshold}"] = val

    if test_file is not None:
        test_data = read_data(test_file)
        logging.info("Training IBM Model 1 on test data with full tags")
        em_segmenter_full = IBM1(num_iterations=iterations)
        em_segmenter_full.train(test_data)
        logging.info("Computing scores for test data with full tags")
        for threshold in thresholds:
            test_scores = compute_score(test_data, em_segmenter_full, threshold)
            for k, val in test_scores.items():
                results[f"test-{k}-{threshold}"] = val

        logging.info("Training IBM Model 1 on test data with split tags")
        em_segmenter_split = IBM1(num_iterations=iterations, split_tags=True)
        em_segmenter_split.train(test_data)
        logging.info("Computing scores for test data with split tags")
        for threshold in thresholds:
            test_scores = compute_score(test_data, em_segmenter_split, threshold)
            for k, val in test_scores.items():
                results[f"test-{k}-{threshold}"] = val
        
        precisions = []
        recalls = []
        f_scores = []
        num_segments = []
        num_segments_ratio = []
        total_characters = 0
        
        logging.info("Evaluating segmentations")
        with open(test_file, 'r', encoding='utf-8') as test_f, \
             open(gold_file, 'r', encoding='utf-8') as gold_f:
                for pred_line, gold_line in zip(test_f, gold_f):
                    pred_parts = pred_line.strip().split('\t')
                    gold_parts = gold_line.strip().split('\t')
                    total_characters += len(pred_parts[0])

                    if len(pred_parts) != 3 or len(gold_parts) != 3:
                        continue
                    pred_segments = pred_parts[2].split('|')
                    gold_segments = gold_parts[2].split('|')

                    pred_idx = boundary_positions(pred_segments)
                    gold_idx = boundary_positions(gold_segments)

                    true_positive = len(gold_idx.intersection(pred_idx))

                    precisions.append(true_positive / (len(pred_idx) + 1e-10))
                    recalls.append(true_positive / (len(gold_idx) + 1e-10))
                    f_scores.append(2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1] + 1e-10))
                    num_segments.append(len(pred_segments))
                    num_segments_ratio.append(len(pred_segments) / len(gold_segments))

        results["boundary_precision"] = np.mean(precisions)
        results["boundary_recall"] = np.mean(recalls)
        results["boundary_f_score"] = np.mean(f_scores)
        results["avg_segments"] = np.mean(num_segments)
        results["chars_per_segment"] = total_characters / np.sum(num_segments)
        results["segments_ratio"] = np.mean(num_segments_ratio)

    logging.info("Done.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IBM Model1 algorithm for extracting a morpheme-aligned score for subword tokens")
    parser.add_argument("--filename", type=str, required=True, help="Path to gold segmentation file")
    parser.add_argument("--test", type=str, required=False, help="Path to predicted segmentation file")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1], help="Probability thresholds")
    parser.add_argument("--iterations", type=int, default=100, help="IBM iterations")

    args = parser.parse_args()

    results = evaluate_segmentations(args.filename, args.test, args.iterations, args.thresholds)
    print(json.dumps(results, indent=4))
