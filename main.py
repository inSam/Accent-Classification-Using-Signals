import sys

import argparse
import pandas as pd
import numpy as np

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='EE269 Project: Audio Classification')
    parser.add_argument('--source_csv', type=str, default="speakers_all.csv",
                        help="Master csv containing audio information")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Classify information
    """
    args = get_args()
    df = pd.read(args.source_csv)
