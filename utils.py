import pandas as pd
import os
import kagglehub

def load_data():
    """
    Loads a CSV file into a pandas DataFrame.
    """

    path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
    data_path = os.path.join(path, 'alzheimers_disease_data.csv')

    return pd.read_csv(data_path)

def column_types(df, tgt_col):
    """
    Separates DataFrame columns into numerical, categorical, binary, and target.
    """
    cat_col = ['Ethnicity']
    nunique = df.nunique()
    bin_col = nunique[nunique == 2].index.tolist()
    num_col = df.columns.drop(tgt_col + cat_col + bin_col)
                         
    return num_col, cat_col, bin_col, tgt_col