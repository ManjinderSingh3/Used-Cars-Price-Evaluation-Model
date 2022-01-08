from pathlib import Path
from typing import List
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


# method to read csv file from given path
def read_csv_file(path: Path):
    return pd.read_csv(path)


# method to write dataframe in given path as csv
def write_csv_file(df: pd.DataFrame, path: Path):
    return df.to_csv(path, index=False)


# This method will be used to get the list of numeric columns
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=['int32', 'float32', 'int64', 'float64']).columns)


# This method will be used to get the list of binary columns
def get_binary_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include='bool').columns)


# This method will be used to get the list of categorical columns
def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include='object').columns)


# method fix_outlier: This method will be used to fixed the outliers in a column
# parameters: df -> dataframe, column -> column name
# returns: dataframe
def fix_outlier(df: pd.DataFrame, column: str):
    df1 = df.copy(deep=True)

    q1 = df1[column].quantile(0.25)
    q3 = df1[column].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (iqr * 1.5)
    upper_limit = q3 + (iqr * 1.5)

    if df1[column].count() > 10000:
        non_outliers = df1[column].between(lower_limit, upper_limit)
        df1 = df1[non_outliers]
    else:
        mean = df1[column].mean()
        df1.loc[df[column] < lower_limit, column] = mean
        df1.loc[df[column] > upper_limit, column] = mean
    return df1


# method fix_nan to fix nan
# parameters: df -> Dataframe, column -> column name
# returns: Dataframe
def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df1 = df.copy(deep=True)
    nan_count = df1[column].isna().sum()
    total_count = df1[column].count()
    if nan_count > (0.05 * total_count):
        df1.dropna(subset=[column], inplace=True)
    else:
        if column in get_numeric_columns(df1):
            df1[column].fillna(df1[column].mean(), inplace=True)
        elif df1[column].dtypes in ['datetime64[ns]']:
            df1[column].fillna(datetime.now().date(), inplace=True)
        elif df1[column].dtypes in ['bool']:
            df1[column].fillna(False, inplace=True)
        elif df1[column].dtypes in ['object']:
            df1[column].fillna(df1[column].mode().values[0], inplace=True)
    return df1


# method to remove data points having invalid characters
def remove_invalid_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df[~df[column].str.contains(r'[!@#$%^&*()-/]')]
    return df


# method to do normalization of column
def normalize_column(df_column: pd.Series) -> pd.Series:
    if df_column.dtypes in ['int32', 'int64', 'float32', 'float64']:
        min_value = df_column.dropna().min()
        max_value = df_column.dropna().max()
        df_column = (df_column - min_value) / (max_value - min_value)
    return df_column


###################################################################
#  Data Encoding Methods
###################################################################

def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    label_encoder = LabelEncoder()
    column_label_encoder = label_encoder.fit(df_column)
    return column_label_encoder


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    df1 = df.copy(deep=True)
    df1[column] = le.transform(df1[column].values)
    return df1


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    df1 = df.copy(deep=True)
    df1[column] = le.inverse_transform(df1[column])
    return df1


def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    df_column = df_column.to_frame()
    column_one_hot_encoder = one_hot_encoder.fit(df_column)
    return column_one_hot_encoder


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str,
                                 ohe: OneHotEncoder, ohe_column_names: List[str]) -> pd.DataFrame:
    df1 = df.copy(deep=True)
    enc_col = pd.DataFrame(ohe.transform(df1[column].values.reshape(-1, 1)).toarray(), columns=ohe_column_names)
    df1 = df1.join(enc_col, rsuffix=f'_{column}')
    df1 = df1.drop([column], axis=1)  # dropping the original column
    return df1


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    df1 = df.copy(deep=True)
    decode_column = pd.DataFrame(ohe.inverse_transform(df1[columns].values).squeeze(), columns=[original_column_name])
    df1 = df1.join(decode_column)
    df1 = df1.drop(columns, axis=1)
    return df1

