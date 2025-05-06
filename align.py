from collections import defaultdict
import argparse
import math
from metrics import get_bin_precision, get_bin_recall, segments_to_binary

parser = argparse.ArgumentParser(description="Run the IBM Model1 algorithm for extracting a morpheme-aligned score for subword tokens")
parser.add_argument("--filename", type=str, required=True, help="Path to gold segmentation file")
parser.add_argument("--test", type=str, required=False, help="Path to predicted segmentation file")
parser.add_argument("--threshold", type=float, default=0.1, help="Probability threshold")
parser.add_argument("--iterations", type=int, default=100, help="IBM iterations")

args = parser.parse_args()

class IBM1:
    def __init__(self, num_iterations=10):
        self.translation_probs = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.num_iterations = num_iterations

    def train(self, data):
        for _ in range(self.num_iterations):
            count_st = defaultdict(lambda: defaultdict(float))
            total_s = defaultdict(float)

            for full_word, tags, segments in data:
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
            lemma, word, tag, segmentation = line.split("\t")
            tags = tag.split("|")
            segments = segmentation.strip().split("|")  
            data.append((word, tags, segments))      
    
    return data

def compute_score(data, em_segmenter, threshold=args.threshold):
    word2score = defaultdict(list)
    for word, tag, segmentation in data:
        for segment in segmentation:
            prob = em_segmenter.get_prob(segment, tag)
            for scores in prob.values():
                if scores > args.threshold:
                    word2score[word].append(scores)

    s =list()
    g = list()
    for word, scores in word2score.items():
        word_score = math.prod(scores)
        geo_mean = (math.pow(word_score, (1 / len(scores))))
        s.append(word_score)
        g.append(geo_mean)

    total_s = sum(s)/ len(s)
    total_g = sum(g)/ len(g)
    
    return total_g

em_segmenter = IBM1(num_iterations=args.iterations)

gold_data = read_data(args.filename)
em_segmenter.train(gold_data)
gold_score = compute_score(gold_data, em_segmenter, args.threshold)
print(f'Gold data morpho-score: {gold_score:.2f}')

if args.test:

    test_data = read_data(args.test)
    em_segmenter.train(test_data)
    test_score = compute_score(test_data, em_segmenter, args.threshold)
    print(f'Test data morpho-score: {test_score:.2f}')
    
    precision = []
    recall = []
    
    with open(args.filename, 'r', encoding='utf-8') as pred_file, \
        open(args.test, 'r', encoding='utf-8') as gold_file:
            for pred_line, gold_line in zip(pred_file, gold_file):
                pred_parts = pred_line.strip().split('\t')
                gold_parts = gold_line.strip().split('\t')

                pred_segments = pred_parts[2].split('|')
                gold_segments = gold_parts[2].split('|')

                pred_idx = segments_to_binary(pred_segments)
                gold_idx = segments_to_binary(gold_segments)

                precision.append(get_bin_precision(gold_idx, pred_idx))
                recall.append(get_bin_recall(gold_idx, pred_idx))


    avg_precision = (sum(precision)/len(precision))
    avg_recall = (sum(recall)/len(recall))

    print(f"Average Boundary Precision: {avg_precision:.2f}")
    print(f"Average Boundary Recall: {avg_recall:.2f}")
    
