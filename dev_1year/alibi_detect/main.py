import datetime
import argparse
from data.process_images import process_images
from data.transform_object import transform_object

from detectors.ClassifierDrift import classifier_drift
from detectors.MMDDrift import mmd_drift
from detectors.KSDrift import ks_drift


def get_args():
    parser = argparse.ArgumentParser(description='CHOOSE MODE')
    parser.add_argument("--mode", "-m", type=str, default="classifier", help="Choose detector type")
    parser.add_argument("--orig_path", "-o", type=str, default="", help="original file path")
    parser.add_argument("--compare_path", "-c", type=str, default="", help="compare file path")
    parser.add_argument("--save", "-s", type=str, default=False, help="save detector")
    parser.add_argument("--load", "-l", type=str, default=False, help="load pretrained detector")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    start_time = datetime.datetime.now()
    train_images, test_images, compare_images = process_images(args.orig_path, args.compare_path)
    X_ref, X_h0, X_c, X_c_names = transform_object(train_images, test_images, compare_images)
    if args.mode == "classifier":
        classifier_drift(X_ref, X_h0, X_c, X_c_names, save=args.save, load=args.load)
    elif args.mode == "mmd":
        mmd_drift(X_ref, X_h0, X_c, X_c_names, save=args.save, load=args.load)
    elif args.mode == "ks":
        ks_drift(X_ref, X_h0, X_c, X_c_names, save=args.save, load=args.load)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    print("================END================")
    print(f"Elapsed time : {elapsed_time}")