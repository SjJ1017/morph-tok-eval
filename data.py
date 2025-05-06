import argparse
from collections import defaultdict


parser = argparse.ArgumentParser(description="Run the IBM Model1 algorithm for extracting a morpheme-aligned score for subword tokens")
parser.add_argument("--features", help='The UniMorph file')
parser.add_argument("--segments", help='The UniMorph file')
args = parser.parse_args()

def unimorph(data):
    word2features = defaultdict()
    with open (data) as file:
        for lines in file:
            lines = lines.split('\t')
            
            #lemma = lines[0]
            form = lines[0]
           
            features = lines[1]
        
            # segmentation = lines[2].strip().replace('|', ' ')
            
            word2features[form] = features
    
    return word2features

def unisegments(data):
    word2segments = defaultdict()
    with open (data) as file:
        for lines in file:
            lines = lines.split('\t')
            form = lines[0]
            segmentaion = lines[3].replace(' + ','|')
            word2segments[form] = segmentaion
    
    return word2segments


features_dict = unimorph(args.features)

segments_dict = unisegments(args.segments)

for forms in features_dict.keys():
    if forms in segments_dict:
        print(forms, features_dict[forms], segments_dict[forms], sep = '\t')



# for forms in features_dict.keys():
#     if forms in segments_dict:
#         print(forms, features_dict[forms], segments_dict[forms], sep = '\t')