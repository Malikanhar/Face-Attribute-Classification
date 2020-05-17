import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from time import gmtime, strftime
import os

def get_model(model, input_shape):
    '''
    Get the base model of the `model` name

    Parameters
    ----------
    model : String representing the base model used (vgg16, inception_v3, or resnet50)
    input_shape  : Input shape for the model, should be (height, width, 3)
    
    Return
    ------
        Model with imagenet weights provided by tensorflow
    '''

    return {
        'vgg16' : tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape),
        'inception_v3' : tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape),
        'resnet50' : tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    }[model]

def build_model(model, input_shape, num_classes):
    '''
    Get the base model of the `model` name

    Parameters
    ----------
    model : String representing the base model used (vgg16, inception_v3, or resnet50)
    input_shape  : Input shape for the model, should be (height, width, 3)
    num_classes  : The number of classes for the output dense
    
    Return
    ------
        Model with the classification head
    '''

    base_model = get_model(model, input_shape)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    # For the multilabel data, we use sigmoid activation instead of softmax
    predictions = Dense(num_classes, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=predictions)

def F1_score(y_true, y_pred):
    '''Define f1 measurement'''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return 2*((precision * recall) / (precision + recall + K.epsilon()))

def define_metrics():
    '''Define the training metrics.'''
    return [
      BinaryAccuracy(name='accuracy'),
      F1_score,
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc'),
    ]

def define_callbacks(ckpt_path, monitor='val_loss'):
    '''Define the training callbacks.'''
    return [
        ModelCheckpoint(ckpt_path, monitor=monitor, verbose=0, save_best_only=True, 
                        save_weights_only=False, mode='auto', save_freq='epoch'),
        ReduceLROnPlateau(monitor=monitor, factor=0.1,
                              patience=6, min_lr=1e-13),
        EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True)
    ]

def save_model(model, model_dir, model_name):
    '''Save the model in model_dir.'''
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_filename = model_name + '_' + strftime("%Y_%m_%d_%H_%M", gmtime()) + '.h5'
    model.save(os.path.join(model_dir, model_filename))
    print('Model saved at {}'.format(model_dir))

def load_pretrained(model_path):
    '''Load pretrained model.'''
    return load_model(model_path)