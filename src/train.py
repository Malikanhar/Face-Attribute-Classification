'''
Train Face Attribute Classification Model
'''

import tensorflow as tf
import argparse
from tensorflow.keras.optimizers import Adam
from utils import load_dataset
from model import build_model, define_metrics, save_model, load_pretrained

def main():
    parser = argparse.ArgumentParser(description='Parser to train Face Attribute Model')
    parser.add_argument('--model', type=str, default='vgg16',
                                    help='Base model to use, vgg16, inception_v3, or resnet50')
    parser.add_argument('--model_dir', type=str, default='models',
                                    help='path to save model')
    parser.add_argument('--pretrained', type=str, default=None,
                                    help='path to pretrained model')
    parser.add_argument('--train_tfrecord', type=str, required=True,
                                    help='Train tfrecord file')
    parser.add_argument('--val_tfrecord', type=str, required=True,
                                    help='Validation tfrecord file')
    parser.add_argument('--img_size', type=int, default=224,
                                    help='Size of image dataset')
    parser.add_argument('--num_classes', type=int, default=4,
                                    help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=64,
                                    help='Size of image dataset')
    parser.add_argument('--epochs', type=int, default=40,
                                    help='Size of image dataset')
    parser.add_argument('--summary', type=bool, default=True,
                                    help='Model summary will be shown if summary is True')

    args = parser.parse_args()
    base_model = args.model
    model_dir = args.model_dir
    train_tfrecord = args.train_tfrecord
    val_tfrecord = args.val_tfrecord
    num_classes = args.num_classes

    input_shape = (args.img_size, args.img_size, 3)
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    num_epochs = args.epochs

    # Load train  and validation data
    train_dataset = load_dataset(train_tfrecord, batch_size, img_size, base_model)
    val_dataset = load_dataset(val_tfrecord, batch_size, img_size, base_model)
    print('Dataset loaded')

    if args.pretrained:
        # Load pretrained model
        attr_model = load_pretrained(args.pretrained)
        print('Model loaded')
    else:
        # Build model
        attr_model = build_model(base_model, input_shape, num_classes)
        print('Model created')

    if args.summary:
        attr_model.summary()

    # Define the training metrics
    metrics = define_metrics()

    # Define the optimizer
    opt = Adam(learning_rate = 1e-4)

    # Compile the model using binary_crossentropy
    attr_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)

    # Start training model
    attr_model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, verbose=1)

    # Save the model
    model_filename = 'attr_model.h5'
    save_model(attr_model, model_dir, base_model)

if __name__ == "__main__":
    main()
