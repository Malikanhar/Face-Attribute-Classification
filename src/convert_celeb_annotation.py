import re
import json
import argparse
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Parser for CelebA Annotation Converter")
    parser.add_argument("--annotation", type=str, required=True,
                                    help="annotation file with .txt extensions")
    parser.add_argument("--json", type=str, default='celeba_annotation.json',
                                    help="output annotation .json filename")

    args = parser.parse_args()

    annotation_path = args.annotation
    json_filename = args.json

    print('Converting CelebA annotation')

    with open(annotation_path) as f:
        num_annotation = int(f.readline()[:-1])
        classes = re.findall(r'\s?(\s*\S+)', f.readline().rstrip())
        selected_classes = [classes.index("Eyeglasses"), classes.index("Mustache"), classes.index("No_Beard"), classes.index("Wearing_Hat")]
        annotations = {}
        pbar = tqdm(total=num_annotation)
        for i in range(num_annotation):
            data = f.readline()
            filename, labels = data.split(" ")[0], re.findall(r'\s?(\s*\S+)', data.rstrip())[1:]
            labels = np.array(list(np.array(labels)[selected_classes]))
            labels[2] = np.where(labels[2] == ' 1', '-1', '1')
            labels = np.where(labels == '-1', '0', '1')
            if '1' in labels:
                annotations[filename] = list(labels)
            pbar.update(1)
    pbar.close()

    with open(json_filename, 'w') as f:
        json.dump(annotations, f)
        
    print('Annotation saved at {}'.format(json_filename))

if __name__ == "__main__":
    main()