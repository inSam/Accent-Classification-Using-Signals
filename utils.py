import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_df(df):
    """
    Remove the unwanted samples from the dataframe
    """
    df = df[(df.native_language == 'english') | (df.length_of_english_residence < 10)]
    return df


def split_data(df, test_split=0.2):
    """
    Creates a train/test split of the data
    """
    return train_test_split(df['language_num'], df['native_language'],
                            test_size=test_split, train_size=(1.0 - test_split),
                            random_state=2840305)
