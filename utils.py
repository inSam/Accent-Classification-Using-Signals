import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def filter_df(df):
    """
    Remove the unwanted samples from the dataframe
    """
    df = df[df['file_missing?'] is False].reset_index()
