import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_categorical_column_names(dataframe: pd.DataFrame) -> list[str]:
    categorical_columns = list(dataframe.select_dtypes(include="object").columns)
    return categorical_columns

def get_continuous_column_names(dataframe: pd.DataFrame) -> list[str]:
    continuous_columns = list(dataframe.select_dtypes(include="number").columns)
    return continuous_columns

def impute_categorical_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    imputed_df = dataframe.copy()
    new_value = "Missing!"
    categorical_columns = imputed_df.select_dtypes(include="object").columns
    for column_name in categorical_columns:
        imputed_df[column_name] = imputed_df[column_name].fillna(new_value)
    return imputed_df

def impute_continuous_missing_values(dataframe: pd.DataFrame, strategy:str = "median") -> pd.DataFrame:
    imputed_df = dataframe.copy()
    
    continuous_columns = imputed_df.select_dtypes(include="number").columns
    
    for column_name in continuous_columns:
        if strategy == "median": 
            new_value = imputed_df[column_name].median()
        imputed_df[column_name] = imputed_df[column_name].fillna(0)
    return imputed_df

def impute_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    encoded_dataframe = impute_continuous_missing_values(dataframe)
    encoded_dataframe = impute_categorical_missing_values(encoded_dataframe)
    return encoded_dataframe

def encode_categorical_data(dataframe: pd.DataFrame, one_hot_encoder: OneHotEncoder) -> pd.DataFrame:
    categorical_columns = get_categorical_column_names(dataframe)
    continuous_columns = get_continuous_column_names(dataframe)
    encoded_categorical_df = pd.DataFrame.sparse.from_spmatrix(one_hot_encoder.transform(dataframe[categorical_columns]),index = dataframe.index , columns=one_hot_encoder.get_feature_names())

    return pd.concat([dataframe[continuous_columns], encoded_categorical_df], axis=1)
    
def get_one_encoder(dataframe: pd.DataFrame) -> sklearn.preprocessing._encoders.OneHotEncoder:
    categorical_columns = get_categorical_column_names(dataframe)
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder.fit(dataframe[categorical_columns])
    return one_hot_encoder

def preprocess(target_column: str, dataframe: pd.DataFrame, encoder: sklearn.preprocessing._encoders.OneHotEncoder = None) -> tuple[pd.DataFrame, sklearn.preprocessing._encoders.OneHotEncoder]:
    preprocessed_dataframe = dataframe.copy()
    if target_column in list(preprocessed_dataframe.columns):
        preprocessed_dataframe = preprocessed_dataframe.drop(columns=[target_column])
    preprocessed_dataframe = impute_missing_values(preprocessed_dataframe)
    
    if(not encoder):
        encoder = get_one_encoder(preprocessed_dataframe)
    preprocessed_dataframe = encode_categorical_data(preprocessed_dataframe, encoder)
    return (preprocessed_dataframe, encoder)
