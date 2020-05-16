# Face-Attribute-Classification
##### Eyeglasses, Beard, Mustache, and Hat classification using CelebA dataset.

## Problem Analysis
The face attribute classification is a multi label classification with four classes for each label including eyeglasses, mustache, beard, and hat. Each class on each label has a binary of 0 or 1 where 0 represents the absence of the class and 1 indicates the existence of the class. For example, if we have a target with values of `1, 0, 0, 1` it means that this face image has the attributes of eyeglasses and hat.

### Transfer Learning
Transfer learning is a pretty good technique to speed up the training process and avoid overfitting due to lack of training data. There are several models such as VGG16, InvceptionV3, Resnet, etc., which have been trained using the imagenet dataset. We can use the pre-trained weight and then freeze some or all of the convolution layers from the model. Lastly, add some Dense layers for the classification head with the `Sigmoid` activation for the last Dense.

### Sigmoid Activation Rather Than Softmax
The idea behind the softmax function is to normalize the data such that the values of the output layer in the network lie in the range of 0 to 1 and the sum of total values equal to 1. These values can then be interpreted as probabilities, where the highest probability is most likely the best candidate label for the sample in the dataset. Of course, this is acceptable for single label data because each label is considered mutually exclusive. For multi-label data another option should be considered. Because we cannot use softmax in this case, we should use some other functions that have a range of 0 or 1, so that these can be interpreted as probabilities. The sigmoid function is a good use for this task. Since the predictions in the output layer of the network are independent of the other output nodes, we can set a threshold to determine the classes for which the sample belongs. In our case, the threshold for the output layer is `0.5`.

## Getting Started
### Data Preparation
First you need to download the CelebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

#### Convert CelebA annotation
Convert CelebA annotations using the following command
<pre>
python src/convert_celeba_annotation.py 
  --annotation annotation.txt 
  --json selected_celeba_annotation.json
</pre>

Now you have the selected_celeba_annotation.json consisting of annotation for the selected classes, in this case we only select the eyeglasses, beard, mustache, dan hat class. In order to make the model can learn the negative class, there is another flag called `mixed_num` that represents the number of annotation for non-selected class to be added.

#### Create tfrecord files

Create tfrecord files for training, validation, and testing with ratio : `80% training`, `10% validation`, and `10% testing`.
<pre>
python src/create_tfrecords.py 
  --json selected_celeba_annotation.json
  --dataset dataset
</pre>

### Training
<pre>
python src/train.py
  --model resnet50
  --model_dir models
  --train_tfrecord train.tfrecord
  --val_tfrecord val.tfrecord
</pre>

By default, this command will start the training for `40` epochs with `1e-4` of learning rate. The model will be trained with transfer learning using the `imagenet` weights provided by Tensorflow and we will freeze the weights so that only classification head will be trained.

### Predict Single Image
<pre>
python predict.py
  --model_path resnet50_2020_05_14_10_45.h5
  --image image.jpg
</pre>

### Download Pre-trained Model
Download pretrained face-attr classification model [here](https://drive.google.com/file/d/1VxXkHyhFqFlrrIKoALEVxmqrrpmT8vSd/view?usp=sharing)

## Result
![Result](https://github.com/Malikanhar/Face-Attribute-Detection/raw/master/assets/result.JPG)
