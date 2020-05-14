'''
Create tfrecords
----------------
Input  : Annotation .json file with the following data structure
         {
             '000001' : ['1', '0', '0', '1'],
             '000002' : ['0', '1', '1', '0'],
             ....
             ....
             ....
             ....
             '202599' : ['0', '0', '1', '1'] 
         }
Output : tfrecord file consisting of encoded image and its label
'''

import tensorflow as tf
import numpy as np
import argparse
import json
import random
import os
from tqdm import tqdm

def _int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    '''Returns a bytes_list from a string / byte.'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_tf_example(example):
    '''
    Create tf.Example data to be serialized in tfrecord

    Parameters
    ----------
    example : A dictionary with encoded and label keys 
              example = {
                'encoded': file,
                'label': label
              }
    Return
    ------
        tf.Example
    '''
    encoded = example['encoded']
    label = example['label']
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(encoded),
        'image/label' : _int64_feature(label)
    }))

def main():
    parser = argparse.ArgumentParser(description='Parser to Create TFrecords')
    parser.add_argument('--json', type=str, required=True,
                                    help='Input annotation filename with .json extensions')
    parser.add_argument('--dataset', type=str, required=True,
                                    help='Path to image dataset')
    parser.add_argument('--img_ext', type=str, default='.png',
                                    help='Image extensions')
    parser.add_argument('--o_train', type=str, default='train.tfrecord')
    parser.add_argument('--o_val', type=str, default='val.tfrecord')
    parser.add_argument('--o_test', type=str, default='test.tfrecord')
    parser.add_argument('--validation', type=float, default=0.1,
                                    help='Validation ratio, with range 0 to 1')
    parser.add_argument('--test', type=float, default=0.1,
                                    help='Validation ratio, with range 0 to 1')

    args = parser.parse_args()
    json_annotation = args.json
    dataset_path = args.dataset
    image_ext = args.img_ext

    # Open .json annotation file
    with open(json_annotation, 'r') as file:
        data = json.load(file)
    
    # Get all filenames from the annotation file
    filenames = list(data.keys())
    random.shuffle(filenames)

    # Create train tfrecord file
    train = tf.io.TFRecordWriter(args.o_train)

    # Create validation tfrecord file
    val = tf.io.TFRecordWriter(args.o_val)

    # Create test tfrecord file
    test = tf.io.TFRecordWriter(args.o_test)

    val_total = int(args.validation * len(filenames))
    val_num = 0

    test_total = int(args.validation * len(filenames))
    test_num = 0
    
    for filename in tqdm(filenames, 'Loading image'):
        file = open(os.path.join(dataset_path, filename) + image_ext, "rb").read()
        label = list(np.array(data[filename]).astype(int))
        example = {
            'encoded': file,
            'label': label
        }
        # Create tf.Example
        tf_example = create_tf_example(example)
        
        # Write serialized tf.Example for validation
        if val_num <= val_total:
            val.write(tf_example.SerializeToString())
            val_num += 1
        elif test_num <= test_total:
            test.write(tf_example.SerializeToString())
            test_num += 1
        # Write serialized tf.Example for train
        else:
            train.write(tf_example.SerializeToString())

    # Close train and validation tfrecord file
    train.close()
    val.close()
    test.close()

if __name__ == "__main__":
    main()