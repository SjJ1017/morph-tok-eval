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
    def __init__(self, num_iterations, split_tags=False):
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        self.num_iterations = num_iterations
        self.split_tags = split_tags
        self.vocabulary_initialized = False
        
    def _initialize_uniform_probabilities(self, data):
        if self.vocabulary_initialized:
            return
            
        all_tags = set()
        all_segments = set()
        
        for _, full_tags, split_tags, segments in data:
            if self.split_tags:
                tags = split_tags
            else:
                tags = full_tags
            all_tags.update(tags)
            all_segments.update(segments)
    
        uniform_prob = 1.0 / len(all_tags) if all_tags else 0.0
        
        for segment in all_segments:
            for tag in all_tags:
                self.translation_probs[tag][segment] = uniform_prob
                
        self.vocabulary_initialized = True

    def train(self, data):
        self._initialize_uniform_probabilities(data)
        
        for iteration in range(self.num_iterations):
            
            count_ts = defaultdict(lambda: defaultdict(float))  
            total_t = defaultdict(float) 
            
            for _, full_tags, split_tags, segments in data:
                if self.split_tags:
                    tags = split_tags
                else:
                    tags = full_tags
                
                for tag in tags:
                    
                    tag_segment_probs = {}
                    norm_factor = 0.0
                    
                    for segment in segments:
                        prob = self.translation_probs[tag][segment]
                        tag_segment_probs[segment] = prob
                        norm_factor += prob
                    
                    norm_factor += 1e-10
                    
                    for segment in segments:
                        delta = tag_segment_probs[segment] / norm_factor
                        count_ts[tag][segment] += delta
                        total_t[tag] += delta
            
            for tag in count_ts:
                total_tag = total_t[tag] + 1e-10  
                for segment in count_ts[tag]:
                    self.translation_probs[tag][segment] = count_ts[tag][segment] / total_tag

    def get_prob(self, segment, tags):
        if isinstance(tags, list):
            return {tag: self.translation_probs[tag][segment] for tag in tags}
        else:
            return self.translation_probs[tags][segment]

class IBM2:
    def __init__(self, num_iterations, split_tags=False):
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        self.alignment_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
        self.num_iterations = num_iterations
        self.split_tags = split_tags
        self.vocabulary_initialized = False
        
    def _initialize_uniform_probabilities(self, data):
        if self.vocabulary_initialized:
            return
            
        all_tags = set()
        all_segments = set()
        
        for _, full_tags, split_tags, segments in data:
            if self.split_tags:
                tags = split_tags
            else:
                tags = full_tags
            all_tags.update(tags)
            all_segments.update(segments)
        
        uniform_trans_prob = 1.0 / len(all_tags) if all_tags else 0.0
        
        for segment in all_segments:
            for tag in all_tags:
                self.translation_probs[tag][segment] = uniform_trans_prob
        
        for _, full_tags, split_tags, segments in data:
            if self.split_tags:
                tags = split_tags
            else:
                tags = full_tags
            
            num_tags = len(tags)
            num_segments = len(segments)
            
            uniform_align_prob = 1.0 / num_segments if num_segments > 0 else 0.0
            
            for tag_pos in range(num_tags):
                for segment_pos in range(num_segments):
                    self.alignment_probs[tag_pos][segment_pos][num_tags][num_segments] = uniform_align_prob
                
        self.vocabulary_initialized = True
        
    def train(self, data):
        self._initialize_uniform_probabilities(data)
        
        for iteration in range(self.num_iterations):
            count_ts = defaultdict(lambda: defaultdict(float))  
            total_t = defaultdict(float)  
            
            count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
            total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
            total_log_likelihood = 0.0
            
            for _, full_tags, split_tags, segments in data:
                if self.split_tags:
                    tags = split_tags
                else:
                    tags = full_tags
                
                num_tags = len(tags)
                num_segments = len(segments)
                
                alignment_matrix = np.zeros((num_tags, num_segments))
                
                for tag_pos, tag in enumerate(tags):
                    normalizer = 0.0
                 
                    for segment_pos, segment in enumerate(segments):
                        trans_prob = self.translation_probs[tag][segment]
                        align_prob = self.alignment_probs[tag_pos][segment_pos][num_tags][num_segments]
                        
                        joint_prob = trans_prob * align_prob
                        alignment_matrix[tag_pos][segment_pos] = joint_prob
                        normalizer += joint_prob
                    
                    if normalizer > 1e-10:
                        alignment_matrix[tag_pos] /= normalizer
                        total_log_likelihood += np.log(normalizer)
                
                for tag_pos, tag in enumerate(tags):
                    for segment_pos, segment in enumerate(segments):
                        alignment_count = alignment_matrix[tag_pos][segment_pos]
                        
                        count_ts[tag][segment] += alignment_count
                        total_t[tag] += alignment_count
                        
                        count_align[tag_pos][segment_pos][num_tags][num_segments] += alignment_count
                        total_align[tag_pos][num_tags][num_segments] += alignment_count
            
            for tag in count_ts:
                total_tag = total_t[tag] + 1e-10
                for segment in count_ts[tag]:
                    self.translation_probs[tag][segment] = count_ts[tag][segment] / total_tag
           
            for tag_pos in count_align:
                for segment_pos in count_align[tag_pos]:
                    for num_tags in count_align[tag_pos][segment_pos]:
                        for num_segments in count_align[tag_pos][segment_pos][num_tags]:
                            total_align_count = total_align[tag_pos][num_tags][num_segments] + 1e-10
                            self.alignment_probs[tag_pos][segment_pos][num_tags][num_segments] = (
                                count_align[tag_pos][segment_pos][num_tags][num_segments] / total_align_count
                            )

    def get_prob(self, segment, tags):
        if isinstance(tags, list):
            return {tag: self.translation_probs[tag][segment] for tag in tags}
        else:
            return self.translation_probs[tags][segment]

MODEL_REGISTRY = {
    "IBM1": IBM1,
    "IBM2": IBM2,
}

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


def evaluate_segmentations(gold_file, test_file, thresholds, iterations, model, skip_gold_train=False):
    results = {}
    ModelClass = MODEL_REGISTRY[model]

    gold_data = read_data(gold_file)
    if not skip_gold_train:
        logging.info("Training {} on gold data with full tags".format(model))
        em_segmenter_full = ModelClass(num_iterations=iterations)
        em_segmenter_full.train(gold_data)
        logging.info("Computing scores for gold data with full tags")
        for threshold in thresholds:
            gold_scores_full = compute_score(gold_data, em_segmenter_full, threshold)
            for k, val in gold_scores_full.items():
                results[f"gold-{k}-{threshold}-{model}"] = val

        logging.info("Training {} on gold data with split tags".format(model))
        em_segmenter_split = ModelClass(num_iterations=iterations, split_tags=True)
        em_segmenter_split.train(gold_data)
        logging.info("Computing scores for gold data with split tags")
        for threshold in thresholds:
            gold_scores_split = compute_score(gold_data, em_segmenter_split, threshold)
            for k, val in gold_scores_split.items():
                results[f"gold-{k}-{threshold}-{model}"] = val

    if test_file is not None:
        test_data = read_data(test_file)
        logging.info("Training {} on test data with full tags".format(model))
        em_segmenter_full = ModelClass(num_iterations=iterations)
        em_segmenter_full.train(test_data)
        logging.info("Computing scores for test data with full tags")
        for threshold in thresholds:
            test_scores = compute_score(test_data, em_segmenter_full, threshold)
            for k, val in test_scores.items():
                results[f"test-{k}-{threshold}-{model}"] = val

        logging.info("Training {} on test data with split tags".format(model))
        em_segmenter_split = ModelClass(num_iterations=iterations, split_tags=True)
        em_segmenter_split.train(test_data)
        logging.info("Computing scores for test data with split tags")
        for threshold in thresholds:
            test_scores = compute_score(test_data, em_segmenter_split, threshold)
            for k, val in test_scores.items():
                results[f"test-{k}-{threshold}-{model}"] = val

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
    parser = argparse.ArgumentParser(description="Run the IBM Model 1 or 2 algorithm for extracting a morpheme-aligned score for subword tokens")
    parser.add_argument("--filename", type=str, required=True, help="Path to gold segmentation file")
    parser.add_argument("--test", type=str, required=False, help="Path to predicted segmentation file")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1], help="Probability thresholds")
    parser.add_argument("--iterations", type=int, default=100, help="IBM iterations")
    parser.add_argument("--model", choices=["IBM1", "IBM2"], default="IBM1", help="Which IBM model to use")
    args = parser.parse_args()
    results = evaluate_segmentations(args.filename, args.test, args.iterations, args.thresholds, args.model)
    print(json.dumps(results, indent=4))
