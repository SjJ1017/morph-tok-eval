import argparse
from collections import defaultdict

def unimorph(data):
    word2features = defaultdict()
    with open (data) as file:
        for lines in file:
            if lines.strip():
                lines = lines.strip().split('\t')
            
                form = lines[1] 
                features = lines[2]
                word2features[form] = features
    
    return word2features

def unisegments(data):
    word2segments = defaultdict()
    with open (data) as file:
        for lines in file:
            lines = lines.split('\t')
            form = lines[0]
            segmentation = lines[3].replace(' + ','|')
            if segmentation.strip(): 
                word2segments[form] = segmentation
           
    return word2segments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create training dataset for morph-tok-eval using UniMorph and UniSegments data"
    )
    parser.add_argument("--features", required=True, help="The UniMorph file")
    parser.add_argument("--segments", required=True, help="The UniMorph or UniSegments file")
    args = parser.parse_args()

    features_dict = unimorph(args.features)
    segments_dict = unisegments(args.segments)

    for forms in features_dict.keys():
        if forms in segments_dict:
            print(forms, features_dict[forms], segments_dict[forms], sep='\t')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
