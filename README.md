# Face-Attribute-Detection
Eyeglasses, Beard, Mustache, and Hat detection using CelebA dataset

## Data Preparation
First you need to download the CelebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

### Convert CelebA annotation
Convert CelebA annotations using the following command
<pre>
python src/convert_celeba_annotation.py 
  --annotation annotation.txt 
  --json selected_celeba_annotation.json
</pre>

Now you have the selected_celeba_annotation.json consisting of annotation for the selected classes, in this case we only select the eyeglasses, beard, mustache, dan hat class. In order to make the model can learn the negative class, there is another flag called `mixed_num` that represents the number of annotation for non-selected class to be added.

### Create tfrecord files

Create tfrecord files for training, validation, and testing with ratio : `80% training`, `10% validation`, and `10% testing`.
<pre>
python src/create_tfrecords.py 
  --json selected_celeba_annotation.json
  --dataset dataset
</pre>

## Training
<pre>
python src/train.py
  --model resnet50
  --model_dir models
  --train_tfrecord train.tfrecord
  --val_tfrecord val.tfrecord
</pre>

By default, this command will start the training for `40` epochs with `1e-4` of learning rate. The model will be trained with transfer learning using the `imagenet` weights provided by Tensorflow and we will freeze the weights so that only classification head will be trained.

## Predict Single Image
<pre>
python predict.py
  --model_path resnet50_2020_05_14_10_45.h5
  --image image.jpg
</pre>

## Result
![Result](https://github.com/Malikanhar/Face-Attribute-Detection/raw/master/assets/result.JPG)
