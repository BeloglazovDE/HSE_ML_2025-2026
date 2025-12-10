"""
Модуль загрузки данных
"""

import pandas as pd
import streamlit as st


def load_example_data() -> pd.DataFrame:
    """
    Загрузка данных из ДЗ
    
    Returns:
        df_train, df_test (DataFrames) 
    """
    try:
        df_train = pd.read_csv(
            './df_train.csv'
        )
        df_test = pd.read_csv(
            "./df_test.csv"
        )
        return df_train, df_test
    except Exception as e:
        st.sidebar.error(f":x: Ошибка загрузки примера: {e}")
        return None, None


def load_uploaded_csv(uploaded_file):
    """
    Загрузка данных из пользовательского CSV файла
    
    Args:
        uploaded_file: объект загруженного файла из st.file_uploader
    
    Returns:
        DataFrame или None если ошибка
    """
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(":white_check_mark: Файл загружен!")
        return df
    except Exception as e:
        st.sidebar.error(f":x: Ошибка загрузки файла: {e}")
        return None
    

import pandas as pd

def validate_train_test_schema(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame
    ):
    """
    Проверка, что train и test совместимы по схеме.

    Args:
        df_train, df_test: DataFrames
    """
    report = {}

    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    only_in_train = sorted(train_cols - test_cols)
    only_in_test = sorted(test_cols - train_cols)
    common_cols = sorted(train_cols & test_cols)

    report["only_in_train"] = only_in_train
    report["only_in_test"] = only_in_test

    dtype_mismatch = {}
    for col in common_cols:
        t_type = df_train[col].dtype
        s_type = df_test[col].dtype
        if t_type != s_type:
            dtype_mismatch[col] = {"train": str(t_type), "test": str(s_type)}
    report["dtype_mismatch"] = dtype_mismatch

    report["is_compatible"] = (
        len(only_in_train) == 0
        and len(only_in_test) == 0
        and len(dtype_mismatch) == 0
    )

    return report


def get_data_summary(df):
    """
    Получение сводной информации о датасете
    
    Returns:
        summary (dict)
    """
    import numpy as np
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    summary = {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
    }
    return summary
