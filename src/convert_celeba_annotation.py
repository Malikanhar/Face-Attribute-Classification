'''
Convert CelebA Annotation
-------------------------
Input  : CelebA annotation .txt file 
Output : Annotation .json file with selected classes such as eyeglasses, mustache, beard, and hat
         The output data type would be a dictionary with image filename as the key and selected classes as the value
         The selected classes is a list of binary with length 4 as the number of selected classes
         An example of the output is 
         {
             '000001' : ['1', '0', '0', '1'],
             '000002' : ['0', '1', '1', '0'],
             ....
             ....
             ....
             ....
             '202599' : ['0', '0', '1', '1'] 
         }
'''

import numpy as np
import re
import os
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Parser for CelebA Annotation Converter')
    parser.add_argument('--annotation', type=str, required=True,
                                    help='Annotation file with .txt extensions')
    parser.add_argument('--json', type=str, default='celeba_annotation.json',
                                    help='Output annotation .json filename')
    parser.add_argument('--mixed_num', type=int, default=10000,
                                    help='The number of non-selected class to be added')

    args = parser.parse_args()

    annotation_path = args.annotation
    json_filename = args.json
    mixed = 0

    print('Converting CelebA annotation')

    with open(annotation_path) as f:
        # Get the total number of annotation
        num_annotation = int(f.readline()[:-1])
        
        # Get classes names
        classes = re.findall(r'\s?(\s*\S+)', f.readline().rstrip())

        # Get the selected class index, we only need 4 classes (eyeglasses, mustache, beard, and hat)
        selected_classes = [classes.index('Eyeglasses'), classes.index('Mustache'), classes.index('No_Beard'), classes.index('Wearing_Hat')]
        
        # Create dictionary to save new annotation
        annotations = {}

        # Use tqdm to report the progress
        pbar = tqdm(total=num_annotation)

        for i in range(num_annotation):
            data = f.readline()

            # Get filename, it's the first element of the line
            filename = os.path.splitext(data.split(" ")[0])[0]
            
            # Get the labels, it's the rest of the row
            labels = re.findall(r'\s?(\s*\S+)', data.rstrip())[1:]

            # Get only selected class index
            labels = np.array(list(np.array(labels)[selected_classes]))

            # Convert no_beard label to beard by reversing the label [-1 to 1] and [1 to -1]
            labels[2] = np.where(labels[2] == ' 1', '-1', '1')

            # Currently, the label is -1 that represents absence of the class, let's convert it to 0
            # and 1 represents the presence of the class
            labels = np.where(labels == '-1', '0', '1')

            # Save the annotation if one of the selected classes has a value of 1
            if '1' in labels:
                annotations[filename] = list(labels)
            # Save the annotation for the non-selected classes as the mixed data
            elif mixed < num_mixed:
                annotations[filename] = list(labels)
                mixed += 1
            pbar.update(1)
    pbar.close()

    # Save annotation as a json
    with open(json_filename, 'w') as f:
        json.dump(annotations, f)
        
    print('Annotation saved at {}'.format(json_filename))

if __name__ == "__main__":
    main()