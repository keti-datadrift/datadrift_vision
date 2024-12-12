import datetime
import argparse
from alibi_library import use_aliab

def get_args():
    parser = argparse.ArgumentParser(description='CHOOSE MODE')
    parser.add_argument('--orig_path', '-o', type=str, default="", help='original file path')
    parser.add_argument('--compare_path', '-c', type=str, default="", help='compare file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    start_time = datetime.datetime.now()
    use_aliab(args.orig_path, args.compare_path)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    print("================END================")
    print(f"Elapsed time : {elapsed_time}")