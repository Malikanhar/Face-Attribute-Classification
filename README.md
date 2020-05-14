# Face-Attribute-Detection
Eyeglasses, Beard, Mustache, and Hat detection using CelebA dataset

## Data Preparation
First you need to download the CelebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

### Convert CelebA annotation
<pre>
python src/convert_celeba_annotation.py 
  --annotation annotation.txt 
  --json selected_celeba_annotation.json
</pre>
Now you have the selected_celeba_annotation.json consisting of annotation for the selected classes, in this case we only select the eyeglasses, beard, mustache, dan hat class
