'''
Predict an image with Face Attribute Classification Model
---------------------------------------------------------
There are four classes that can be predicted by the model (Eyeglasses, Mustache, Beard, and Hat)
Before predicting an image, make sure that you have a trained model
Use the preprocessing method according to the model you are using by setting the preprocessing flag
'''

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from utils import transform_image

def convertAttribute(attributes, threshold = 0.5):
    '''
    Converting an array of probability to its corresponding label

    Parameters
    ----------
    attributes : 1D numpy array with length 4 which have values in the range of 0 to 1

    Return
    ------
        label : Dictionary with key as the label and value as the confidence level in percentage
    '''

    attribute_names = ['Eyeglasses', 'Mustache', 'Beard', 'Hat']
    labels = {}
    for i, attr in enumerate(attribute_names):
        # If the probability of a class is more than the threshold then it will be set with its appropriate label
        if attributes[i] > threshold:
            labels[attribute_names[i]] = '{:.1%}'.format(attributes[i])
    return labels

def draw_outputs(img, labels):
    '''
    Put the label to the image

    Parameters
    ----------
    img : numpy array with size of (224, 224)
    label : Dictionary with key as the label and value as the confidence level in percentage

    Return
    ------
    numpy array of an image with its label
    '''
    x1y1 = (10, 224)
    y_txt1 = 10
    y_txt2 = 5
    for key in labels.keys():
        text = '{} {}'.format(key, labels[key])
        color = (0, 255, 255)
        img = cv2.rectangle(img, (x1y1[0] - 5, x1y1[1] - (y_txt1 + 11)), (x1y1[0] + (len(key) * 8) + 40, x1y1[1] - y_txt2), color, -1)
        img = cv2.putText(img, text, (x1y1[0], x1y1[1] - y_txt1), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)
        y_txt1 += 20
        y_txt2 += 20
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description='Parser for Predicting Face Attribute')
    parser.add_argument('--model_path', type=str, required=True,
                                    help='Path to model with .h5 extensions')
    parser.add_argument('--image', type=str, required=True,
                                    help='Image filename to be predicted')
    parser.add_argument('--preprocessing', type=str, default='resnet50',
                                    help='Preprocessing method to be applied for the image, vgg16, inception_v3, or resnet50')
    parser.add_argument('--img_size', type=int, default=224,
                                    help='Image filename to be predicted')
    parser.add_argument('--save_image', type=bool, default=True,
                                    help='Image filename to be predicted')

    args = parser.parse_args()

    model_path = args.model_path
    preprocessing = args.preprocessing
    filename = args.image
    size = (args.img_size, args.img_size)

    # Load image
    img = np.array(load_img(filename))

    # Preprocess the image and expand its dimension
    trans_img = transform_image(img, size, preprocessing)
    trans_img = np.expand_dims(trans_img, 0)

    # Load model and predict the image
    model = load_model(model_path)
    pred = model.predict(trans_img)[0]
    
    # Convert the output of predict to the label
    label = convertAttribute(pred)
    print(label)

    # Save the output image
    if args.save_image:
        img = draw_outputs(cv2.resize(img, size), label)
        cv2.imwrite('out_' + filename, img)
        print('Output image saved at out_{}'.format(filename))

if __name__ == "__main__":
    main()