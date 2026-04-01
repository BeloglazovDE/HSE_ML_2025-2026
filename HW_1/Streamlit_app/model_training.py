"""
Модуль обучения моделей
"""

import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    OrdinalEncoder
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, accuracy_score, 
    classification_report
)


def prepare_data(df_train, df_test, model_config):
    """
    Подготовка данные для обучения
    
    Args:
        df_train, df_test: DataFrame
        model_config: словарь с параметрами модели
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    target_feature = model_config['target_col']
    random_state = model_config['random_state']

    if df_test is None:
        if model_config['mode_na'] == 'Удалить':
            df_train = df_train.dropna()
        X_train_test = df_train.drop(columns=[target_feature], axis=1)
        y_train_test = df_train[target_feature]

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_test,
            y_train_test,
            test_size=model_config['test_size'], 
            random_state=random_state
        )
    else:
        if model_config['mode_na'] == 'Удалить':
            df_train = df_train.dropna()
            df_test = df_test.dropna()
        X_train = df_train.drop([target_feature], axis=1)
        X_test = df_test.drop([target_feature], axis=1)
        y_train = df_train[target_feature]
        y_test = df_test[target_feature]

    num_cols = list(X_train.select_dtypes(include=['int', 'float']).columns)
    cat_cols = list(X_train.select_dtypes(include=['object']).columns)

    transformers = []

    if len(num_cols) > 0:
        numeric_transformer = Pipeline(steps=[])
        
        if model_config['mode_na'] == 'Заполнить':
            numeric_transformer.steps.append(
                ('imputer', SimpleImputer(strategy='median'))
            )
        
        if model_config['normalize']:
            numeric_transformer.steps.append(
                ('scaler', StandardScaler())
            )
        
        transformers.append(
            ('num', numeric_transformer, num_cols)
        )

    if model_config['cat_coding'] and len(cat_cols) > 0:
        ohe_cols = []
        ord_cols = []
        count_cols = []

        for col in cat_cols:
            unique_number = X_train[col].nunique()

            if unique_number <= 20:
                ohe_cols.append(col)
            elif unique_number <= 100:
                ord_cols.append(col)
            else:
                count_cols.append(col)

        if model_config['mode_na'] == 'Заполнить':
            impute_transformer = SimpleImputer(strategy='most_frequent')
            cat_imputed = impute_transformer.fit_transform(X_train[cat_cols])
            X_train_cat = pd.DataFrame(
                cat_imputed, columns=cat_cols, index=X_train.index
            )
            cat_imputed_test = impute_transformer.transform(X_test[cat_cols])
            X_test_cat = pd.DataFrame(
                cat_imputed_test, columns=cat_cols, index=X_test.index
            )
            X_train[cat_cols] = X_train_cat
            X_test[cat_cols] = X_test_cat

        cat_transformers = []

        if ohe_cols:
            cat_transformers.append((
                'ohe', 
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                ohe_cols
            ))

        if ord_cols:
            cat_transformers.append((
                'ordinal',
                OrdinalEncoder(handle_unknown='use_encoded_value', 
                               unknown_value=-1),
                ord_cols
            ))

        if count_cols:
            cat_transformers.append((
                'count',
                ce.CountEncoder(cols=count_cols),
                count_cols
            ))

        if cat_transformers:
            cat_transformer = ColumnTransformer(
                cat_transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            transformers.append(('cat', cat_transformer, cat_cols))
    
    if len(transformers) > 0:
        final_transformer = ColumnTransformer(
            transformers,
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        X_train = final_transformer.fit_transform(X_train)
        X_test = final_transformer.transform(X_test)

        feature_names = final_transformer.get_feature_names_out()
    else:
        feature_names = X_train.columns.to_list()
    
    return X_train, X_test, y_train, y_test, None, feature_names


def model_training_func(
        X_train, y_train, X_test, y_test,
        model_config
    ):
    """
    Обучение выбранной модели
    
    Args:
        model_name (str): название модели 
        X_train, X_test: обучающие и тестовые признаки
        y_train, y_test: обучающие и тестовые целевые переменные
    
    Returns:
        model: обученная модель
        metrics (dict): словарь с метриками на тестовом наборе
    """
    model_name = model_config['model_choice']

    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RandomForestRegression':
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=model_config['random_state'], 
            n_jobs=-1
        )
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(
            max_iter=1000, 
            random_state=model_config['random_state'], 
            n_jobs=-1
        )
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=model_config['random_state'], 
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {}
    
    if model_name in ['LinearRegression', 'RandomForestRegression']:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'rmse': rmse,
            'r2_score': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    else:
        accuracy = accuracy_score(y_test, y_pred)
        n_classes = len(np.unique(y_test))
        
        report_dict = classification_report(
            y_test, y_pred, 
            output_dict=True, 
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        
        metrics = {
            'accuracy': accuracy,
            'n_classes': n_classes,
            'report_df': report_df,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return model, metrics


def make_prediction(model, X_new, scaler=None):
    """
    Сделать предсказание на новых данных
    
    Args:
        model: обученная модель
        X_new: новые данные (DataFrame или array)
        scaler: scaler для нормализации (если использовался)
    
    Returns:
        predictions
    """
    if scaler is not None:
        X_new = scaler.transform(X_new)
    
    predictions = model.predict(X_new)
    return predictions


def get_model_weights(model):
    """
    Получить веса/коэффициенты модели
    
    Args:
        model: обученная модель
    
    Returns:
        weights, has_coef (bool), has_importances (bool)
    """
    weights = None
    has_coef = hasattr(model, 'coef_')
    has_importances = hasattr(model, 'feature_importances_')
    
    if has_coef:
        weights = model.coef_
    elif has_importances:
        weights = model.feature_importances_
    
    return weights, has_coef, has_importances
