from utils import load_dataset
from model import define_metrics, load_pretrained
import argparse

def main():
    parser = argparse.ArgumentParser(description='Parser to evaluate Face Attribute Model')
    parser.add_argument('--mode', type=str, default='Test',
                                    help='Evaluation mode')
    parser.add_argument('--trained_model', type=str, required=True,
                                    help='Face-attr model to be evaluated')
    parser.add_argument('--preprocessing', type=str, default='resnet50',
                                    help='Preprocessing method to be applied for the image, vgg16, inception_v3, or resnet50')
    parser.add_argument('--test', type=str, default='test.tfrecord',
                                    help='Test tfrecord file')
    parser.add_argument('--batch_size', type=int, default=32,
                                    help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                                    help='Batch size')

    args = parser.parse_args()
    # Load Test data
    test_dataset = load_dataset(args.test, args.batch_size, (args.img_size, args.img_size), args.preprocessing)

    # Load model    
    attr_model = load_pretrained(args.trained_model)

    # Define the training metrics
    metrics = define_metrics()

    # Compile the model using binary_crossentropy
    attr_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

    scores = attr_model.evaluate(test_dataset)

    print('Test loss      : {:.2%}'.format(scores[0]))
    print('Test accuracy  : {:.2%}'.format(scores[1]))
    print('Test f1        : {:.2%}'.format(scores[2]))
    print('Test precision : {:.2%}'.format(scores[3]))
    print('Test recall    : {:.2%}'.format(scores[4]))
    print('Test AUC       : {:.2%}'.format(scores[5]))

if __name__ == "__main__":
    main()