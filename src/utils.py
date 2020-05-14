import os
import tensorflow as tf

def transform_image(image, size, model):
    '''
    Preprocesses a tensor or Numpy array encoding a batch of images

    Parameters
    ----------
    image : A Tensor of type uint8
    size  : Image size of type tuple, should be (height, width)
    model : String representing the base model used (vgg16, inception_v3, or resnet50)
    
    Return
    ------
        Preprocessed numpy.array or a tf.Tensor with type float32
        The inputs pixel values are scaled between -1 and 1, sample-wise
    '''

    image = tf.image.resize(image, size)
    return {
        'vgg16' : tf.keras.applications.vgg16.preprocess_input(image),
        'inception_v3' : tf.keras.applications.inception_v3.preprocess_input(image),
        'resnet50': tf.keras.applications.resnet.preprocess_input(image),
    }[model]

def parse_data(example_proto, size, model):
    '''
    Parse tfrecord data

    Parameters
    ----------
    example_proto : A Dataset comprising records from one or more TFRecord files
    size  : Image size of type tuple, should be (height, width)
    model : String representing the base model used (vgg16, inception_v3, or resnet50)
    
    Return
    ------
        x_train : Preprocessed numpy.array or a tf.Tensor with type float32
        y_train : A dense tensor with shape of (length(label), 1)
    '''

    image_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/label': tf.io.VarLenFeature(tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, image_feature_description)
    x_train = tf.image.decode_jpeg(x['image/encoded'])
    x_train = transform_image(x_train, size, model)
    y_train = tf.sparse.to_dense(x['image/label'])
    return x_train, y_train

def load_dataset(path, batch_size, size, model):
    '''
    Load a tfrecord file

    Parameters
    ----------
    path  : Path of tfrecord file
    batch_size : Number of batch size to load the data
    size  : Image size of type tuple, should be (height, width)
    model : String representing the base model used (vgg16, inception_v3, or resnet50)
    
    Return
    ------
        An Iterator over the elements of the dataset
    '''

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(lambda x: parse_data(x, size, model))
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)