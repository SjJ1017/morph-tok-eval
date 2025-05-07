from collections import defaultdict
import argparse
import json
import math
import numpy as np

from metrics import get_bin_precision, get_bin_recall, segments_to_binary


class IBM1:
    def __init__(self, num_iterations=10):
        self.translation_probs = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.num_iterations = num_iterations

    def train(self, data):
        for _ in range(self.num_iterations):
            count_st = defaultdict(lambda: defaultdict(float))
            total_s = defaultdict(float)

            for _, tags, segments in data:
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
            tags = tag.split("|")
            segments = segmentation.strip().split("|")  
            data.append((word, tags, segments))      
    
    return data


def geometric_mean(numbers):
    return np.exp(np.log(numbers).mean())


def harmonic_mean(numbers):
    return len(numbers) / np.sum(1.0 / np.array(numbers))


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def compute_score(data, em_segmenter, threshold):
    results = defaultdict(float)
    for word, tag, segmentation in data:
        for segment in segmentation:
            prob = em_segmenter.get_prob(segment, tag)
            scores = [x for x in prob.values() if x > threshold]
            if len(scores) > 0:
                results['morpho-score-mean'] += sum(scores) / len(scores)
                results['morpho-score-harmonic'] += harmonic_mean(scores)
                results['morpho-score-geometric'] += geometric_mean(scores)
                results['morpho-score-min'] += min(scores)
                results['morpho-score-max'] += max(scores)
                results['morpho-score-entropy'] += entropy(scores)

    for k in results:
        results[k] /= len(data)
    
    return results


def evaluate_segmentations(iterations, threshold, gold_file, test_file):
    em_segmenter = IBM1(num_iterations=iterations)
    results = {}

    gold_data = read_data(gold_file)
    em_segmenter.train(gold_data)
    gold_scores = compute_score(gold_data, em_segmenter, threshold)
    for k, val in gold_scores.items():
        results[f"gold-{k}"] = val

    if test_file is not None:
        test_data = read_data(test_file)
        em_segmenter.train(test_data)
        test_scores = compute_score(test_data, em_segmenter, threshold)
        for k, val in test_scores.items():
            results[f"test-{k}"] = val
        
        precisions = []
        recalls = []
        f_scores = []
        num_segments = []
        
        with open(gold_file, 'r', encoding='utf-8') as pred_file, \
             open(test_file, 'r', encoding='utf-8') as gold_file:
                for pred_line, gold_line in zip(pred_file, gold_file):
                    pred_parts = pred_line.strip().split('\t')
                    gold_parts = gold_line.strip().split('\t')

                    pred_segments = pred_parts[2].split('|')
                    gold_segments = gold_parts[2].split('|')

                    pred_idx = segments_to_binary(pred_segments)
                    gold_idx = segments_to_binary(gold_segments)

                    precisions.append(get_bin_precision(gold_idx, pred_idx))
                    recalls.append(get_bin_recall(gold_idx, pred_idx))
                    f_scores.append(2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1] + 1e-10))
                    num_segments.append(len(pred_segments))

        results["boundary_precision"] = np.mean(precisions)
        results["boundary_recall"] = np.mean(recalls)
        results["boundary_f_score"] = np.mean(f_scores)
        results["avg_segments"] = np.mean(num_segments)

    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IBM Model1 algorithm for extracting a morpheme-aligned score for subword tokens")
    parser.add_argument("--filename", type=str, required=True, help="Path to gold segmentation file")
    parser.add_argument("--test", type=str, required=False, help="Path to predicted segmentation file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Probability threshold")
    parser.add_argument("--iterations", type=int, default=100, help="IBM iterations")

    args = parser.parse_args()

    results = evaluate_segmentations(args.iterations, args.threshold, args.filename, args.test)
    print(json.dumps(results, indent=4))