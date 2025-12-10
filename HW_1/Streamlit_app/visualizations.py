"""
Модуль с фугкциями для отрисовки графиков
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_distribution(
        df, 
        feature,
        hue_target=None,
        max_length: int = 100000,
        hue_nunique_threshold: int = 5,
        discrete_threshold: int = 20,
    ):
    """
    Построение распределения признака
    
    Args:
        df: DataFrame
        feature: название признака
        hue_target: название группы (если есть)
        max_length: максимальная длина для отрисовки
        hue_nunique_threshold: порог уникальных значений для групп
        discrete_threshold: порог дискретности для категорий
    
    Returns:
        fig (plotly figure)
    """
    data = df.sample(min(max_length, df.shape[0]), random_state=42)

    # Группировка по hue_target если слишком много категорий
    if hue_target:
        other_groups = (
            data[hue_target].value_counts()[hue_nunique_threshold:]
        ).index
        if len(other_groups) > 0:
            data = data.copy()
            data[hue_target] = data[hue_target].map(
                {other_group: "other" for other_group in other_groups}
            )

    # Для числового признака
    if (df[feature].dtype in ['int64', 'float64']) and \
        (df[feature].nunique() > discrete_threshold):
        
        hist_fig = px.histogram(
            data, 
            x=feature, 
            nbins=50,
            color=hue_target,
            marginal="box",
            title=f"Гистограмма распределения {feature}",
            labels={feature: feature},
            hover_data={feature: ':.2f'}
        )
        
        hist_fig.update_layout(height=500)
        return hist_fig

    # Для категориального признака
    elif (df[feature].dtype == 'object') and \
        (df[feature].nunique() <= discrete_threshold):
        
        bar_fig = px.bar(
            data,
            y=feature,
            orientation='h',
            title=f"Барплот для {feature}",
            labels={feature: feature},
            color=hue_target
        )
        
        bar_fig.update_layout(height=500, showlegend=True)
        return bar_fig

    else:
        return None



def plot_scatterplot(
        df, 
        x, 
        y, 
        hue_target=None, 
        max_length: int = 100000,
        hue_nunique_threshold: int = 5,
    ):
    """
    Построение диаграммы рассеяния
    
    Args:
        df: DataFrame
        x, y: названия признаков по осям
        hue_target: название группы (если есть)
        max_length: максимальная длина для отрисовки
        hue_nunique_threshold: порог уникальных значений для групп
    
    Returns:
        fig (plotly figure)
    """
    data = df.sample(min(max_length, df.shape[0]), random_state=42)
    
    if hue_target:
        other_groups = (
            data[hue_target].value_counts()[hue_nunique_threshold:]
        ).index
        if len(other_groups) > 0:
            data = data.copy()
            data[hue_target] = data[hue_target].map(
                {other_group: "other" for other_group in other_groups}
            )
    
    fig = px.scatter(
        data, 
        x=x, 
        y=y, 
        color=hue_target,
        title=f"Диаграмма рассеяния {x} и {y}",
        labels={x: x, y: y},
        hover_data={x: ':.2f', y: ':.2f'},
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(height=500)
    
    return fig



def plot_correlation_matrix(df, numeric_cols):
    """
    Построение матрицы корреляции 
    
    Args:
        df: DataFrame
        numeric_cols: список числовых столбцов
    
    Returns:
        fig (plotly figure)
    """
    corr_matrix = df[numeric_cols].corr(method='spearman')
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Feature", y="Feature", color="Корреляция"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.2f',
        aspect='auto',
        title='Матрица корреляции признаков (Спирмен)'
    )
    
    fig.update_layout(
        font=dict(size=10)
    )
    
    return fig



def plot_missing_values(df):
    """
    Построение графика пропущенных значений
    
    Args:
        df: DataFrame
    
    Returns:
        fig (plotly figure)
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    missing_df = pd.DataFrame({
        'Feature': missing_data.index,
        'Missing Count': missing_data.values
    })
    
    fig = px.bar(
        missing_df,
        x='Feature',
        y='Missing Count',
        title='Пропущенные значения по признакам',
        labels={'Feature': 'Признак', 'Missing Count': 'Количество пропусков'},
        color='Missing Count',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig



def plot_prediction_vs_actual(y_test, y_pred):
    """
    Построить график предсказаний vs реальных значений (для регрессии)
    
    Args:
        y_test: реальные значения
        y_pred: предсказанные значения
    
    Returns:
        fig (plotly figure)
    """
    pred_df = pd.DataFrame({
        'Реальные': y_test,
        'Предсказанные': y_pred,
        'Ошибка': y_test - y_pred
    })
    
    fig = px.scatter(
        pred_df,
        x='Реальные',
        y='Предсказанные',
        title='Предсказания vs Реальные значения',
        labels={'Реальные': 'Реальные значения', 
                'Предсказанные': 'Предсказанные значения'},
        opacity=0.6
    )
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(height=500)
    fig.update_traces(marker=dict(size=6))
    
    return fig


def plot_errors(y_test, y_pred):
    """
    Построить график ошибки
    
    Args:
        y_test: реальные значения
        y_pred: предсказанные значения
    
    Returns:
        fig (plotly figure)
    """
    pred_df = pd.DataFrame({
        'Реальные': y_test,
        'Ошибка': y_test - y_pred
    })
    
    fig = px.scatter(
        pred_df,
        x='Реальные',
        y='Ошибка',
        title='График разброса ошибки',
        labels={'Реальные': 'Реальные значения', 
                'Ошибка': 'y_true - y_pred'},
        opacity=0.6
    )
    
    fig.update_layout(height=500)
    fig.update_traces(marker=dict(size=6))
    
    return fig


def plot_coefficients(feature_names, coefficients):
    """
    Построить график коэффициентов модели
    
    Args:
        feature_names: список названий признаков
        coefficients: значения коэффициентов
    
    Returns:
        fig (plotly figure), coef_df
    """
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    coef_df['Coefficient_text'] = coef_df['Coefficient'].apply(
        lambda x: f'{x:.3f}'
    )
    
    fig = px.bar(
        coef_df,
        y='Feature',
        x='Coefficient',
        orientation='h',
        title='Коэффициенты модели',
        labels={'Feature': 'Признак', 'Coefficient': 'Значение коэффициента'},
        color='Coefficient',
        color_continuous_scale='RdYlGn',
        text='Coefficient_text'
    )
    
    fig.update_layout(
        height=max(400, len(feature_names) * 20),
        showlegend=False
    )
    
    fig.update_traces(textposition='outside')
    
    # Добавляем вертикальную линию в 0
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
    
    return fig, coef_df



def plot_feature_importances(feature_names, importances):
    """
    Построить график важности признаков (для Random Forest)
    
    Args:
        feature_names: список названий признаков
        importances: значения важности
    
    Returns:
        fig (plotly figure), importance_df
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    importance_df['Importance_text'] = importance_df['Importance'].apply(
        lambda x: f'{x:.4f}'
    )
    
    fig = px.bar(
        importance_df,
        y='Feature',
        x='Importance',
        orientation='h',
        title='Важность признаков',
        labels={'Feature': 'Признак', 'Importance': 'Importance'},
        color='Importance',
        color_continuous_scale='Blues',
        text='Importance_text'
    )
    
    fig.update_layout(
        height=max(400, len(feature_names) * 20),
        showlegend=False
    )
    
    fig.update_traces(textposition='outside')
    
    return fig, importance_df