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
from mtcnn.mtcnn import MTCNN

def detect_face(img):
    '''
    Face detection using MTCNN

    Parameters
    ----------
    img : 3D numpy array with shape (height, width, channel)

    Return
    ------
        faces  : List of face images that have been cropped according to their bbox
        bboxes : List of bbox with values of x, y, w, h
    '''

    faces = []
    bboxes = []
    detector = MTCNN()
    detected_faces = detector.detect_faces(img)

    for detected_face in detected_faces:
        bbox = detected_face['box']
        
        x_offset = int(bbox[2] * 0.2)
        y_offset = int(bbox[3] * 0.125)

        x1 = max(0, bbox[0] - x_offset)
        y1 = max(0, (bbox[1] - int(2 * y_offset)))
        x2 = min(img.shape[1], bbox[0] + bbox[2] + x_offset)
        y2 = min(img.shape[0], bbox[1] + bbox[3] + y_offset)
        
        faces.append(img[y1 : y2, x1 : x2])
        bboxes.append(bbox)

    return faces, bboxes

def convert_attribute(bboxes, predictions):
    '''
    Converting an array of probability to its corresponding label

    Parameters
    ----------
    bboxes      : List of bbox with values of x, y, w, h
    predictions : 2D numpy array with shape (n, 4) which have values in the range of 0 to 1
                  n is the number of faces detected by MTCNN

    Return
    ------
        outputs : List of dictionary consisting of bbox and its probability of each classes
                example => [{
                    'bbox' : [x, y, w, h]
                    'classes' : {
                        'Eyeglassess' : eyeglasses_confident,
                        'Mustache' : mustache_confident,
                        'Beard' : beard_confident,
                        'Hat' : hat_confident
                    }
                }]
    '''

    attribute_names = ['Eyeglasses', 'Mustache', 'Beard', 'Hat']
    outputs = []
    for i, bbox in enumerate(bboxes):
        output = {}
        classes = {}
        prediction = predictions[i]
        for j in range(len(attribute_names)):
            classes[attribute_names[j]] = prediction[j]
        outputs.append({
            'bbox' : bbox,
            'classes' : classes
        })
    return outputs

def draw_outputs(img, annotations, threshold = 0.5, height = 480):
    '''
    Put the label to the image

    Parameters
    ----------
    img : numpy array with size of (224, 224)
    annotations : List of dictionary consisting of bbox and its probability of each classes

    Return
    ------
        numpy array of resized image with labels for each detected faces
    '''

    color = (0, 255, 255)

    ratio = img.shape[1] / img.shape[0]
    width = int(height * ratio)
    res_img = cv2.resize(img, (width, height))
    
    for annotation in annotations:
        bbox = annotation['bbox']
        label = annotation['classes']

        x1y1 = (int(bbox[0] * height / img.shape[0]), int(bbox[1] * width / img.shape[1]))
        x2y2 = (int((bbox[0] + bbox[2]) * height / img.shape[0]), int((bbox[1] + bbox[3]) * width / img.shape[1]))
        y_txt1 = 10
        y_txt2 = 5

        res_img = cv2.rectangle(res_img, x1y1, x2y2, color, 2)
        
        for key in label.keys():
            # If the probability of a class is more than the threshold then it will be set with its appropriate label
            if label[key] > threshold:
                text = '{} {:.1%}'.format(key, label[key])
                res_img = cv2.rectangle(res_img, (x1y1[0], x1y1[1] - y_txt1 - 11), (x1y1[0] + (len(key) * 8) + 60, x1y1[1] - y_txt2), color, -1)
                res_img = cv2.putText(res_img, text, (x1y1[0] + 5, x1y1[1] - y_txt1), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)
                y_txt1 += 20
                y_txt2 += 20
    return cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

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

    results = None

    # Load model
    model = load_model(model_path)

    # Load image
    img = np.array(load_img(filename))

    # Detect faces using MTCNN face detector
    faces, bboxes = detect_face(img)

    if faces is not None:
        # Preprocess the image and expand its dimension
        trans_imgs = [transform_image(face, size, preprocessing) for face in faces]

        # Predict face attributes
        preds = model.predict(np.array(trans_imgs))

        # Convert the output of predict and combine with its corresponding bbox
        results = convert_attribute(bboxes, preds)

        # Save the output image
        if args.save_image:
            img = draw_outputs(img, results)
            cv2.imwrite('out_{}'.format(filename), img)
            print('Output image saved at out_{}'.format(filename))

    print(results)

if __name__ == "__main__":
    main()