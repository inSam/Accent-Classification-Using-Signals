import logging
import multiprocessing
import pickle

import librosa
import pandas as pd
import numpy as np

from utils import filter_df, split_data
from argparse import ArgumentParser
from functools import partial


def get_args():
    parser = ArgumentParser(description='EE269 Project: Audio Classification')
    parser.add_argument('--source_csv', type=str, default="speakers_scrapped.csv",
                        help="Master csv containing audio information")
    parser.add_argument('--debug', type=bool, default=False, help="Turn on debugging")
    parser.add_argument('--sampling_rate', type=int, default=22050, help="Audio sampling rate")
    parser.add_argument('--load_cache', type=bool, default=True, help="Using cachced MFCC dump so we don't have to re-read the audio files")
    args = parser.parse_args()
    return args


def get_audio(language_num, sr):
    return librosa.load('./audio/{}.wav'.format(language_num), sr)[0]


def to_mfcc(audio, sr):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=int(sr * 25/1000),
                                win_length=int(sr * 25/1000 / 4))


if __name__ == '__main__':
    """
    Classify Audio
    """
    args = get_args()

    if args.load_cache:
        df = pd.read_csv(args.source_csv)
        df = filter_df(df)
        factor = pd.factorize(df['native_language'])
        df['native_language'] = factor[0]
        language_set = factor[1].values
        del factor

        train_X, test_X, train_Y, test_Y = split_data(df, test_split=0.2)
        train_Y, test_Y = train_Y.values, test_Y.values
        train_size, test_size = len(train_X), len(test_X)
        pickle.dump([train_Y, test_Y], open('train_labels.dump', 'wb'))

        logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
        logger = logging.getLogger(__name__)
        logger.debug(" Training set size = {}".format(train_size))
        logger.debug(" Testing set size = {}".format(test_size))
        logger.info(" list of accents: {}".format(language_set))

        logger.info(" Loading audio files...")
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        train_X = pool.map(partial(get_audio, sr=args.sampling_rate), train_X)
        test_X = pool.map(partial(get_audio, sr=args.sampling_rate), test_X)

        logger.info(" Generating MFCC...")
        train_X = pool.map(partial(to_mfcc, sr=args.sampling_rate), train_X)
        test_X = pool.map(partial(to_mfcc, sr=args.sampling_rate), test_X)
        pickle.dump(train_X, open('train_mfcc.dump', 'wb'))
        pickle.dump(test_X, open('test_mfcc.dump', 'wb'))
    else:
        train_Y, test_Y = pickle.load(open('train_labels.dump', 'rb'))
        train_X = pickle.load(open('train_mfcc.dump', 'rb'))
        test_X = pickle.load(open('test_mfcc.dump', 'rb'))
