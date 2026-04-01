"""
Модуль с компонентами пользовательского интерфейса
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from data_loader import get_data_summary, validate_train_test_schema
from visualizations import (
    plot_distribution, 
    plot_scatterplot, 
    plot_correlation_matrix, 
    plot_missing_values,
    plot_prediction_vs_actual,
    plot_errors
)
from model_training import model_training_func

def render_header() -> None:
    """Отрисовка заголовка приложения"""
    st.title(":robot: ML Service")
    st.markdown("Интерактивное приложение для машинного обучения")


def render_data_loading_sidebar():
    """
    Отрисовка боковой панели для загрузки данных
    
    Returns:
        df_train, df_test, data_source, success (bool)
    """
    df_train = None
    df_test = None
    success = False
    has_test = False
    summary = {}

    st.sidebar.header(":bar_chart: Загрузка данных")
    
    data_source = st.sidebar.radio(
        "Выберите источник данных:", 
        ["Пример (Cars)", "Загрузить **CSV**"]
    )
    
    if data_source == "Пример (Cars)":
        from data_loader import load_example_data
        df_train, df_test = load_example_data()
        success = (df_train is not None and df_test is not None)
        if success:
            st.sidebar.success(":white_check_mark: Данные загружены!")
            summary = get_data_summary(df_train)
    
    elif data_source == "Загрузить **CSV**":
        has_test = st.sidebar.checkbox(
            "Есть тестовые данные",
            value=False
        )
        df_train = st.sidebar.file_uploader(
            "Загрузите CSV-файл с :red[тренировочными] данными", 
            type=['csv']
        )
        if df_train is not None:
            from data_loader import load_uploaded_csv
            df_train = load_uploaded_csv(df_train)
            summary = get_data_summary(df_train)
        if has_test:
            df_test = st.sidebar.file_uploader(
                "Загрузите CSV-файл с :red[тестовыми] данными", 
                type=['csv']
            )
            if df_test is not None:
                from data_loader import load_uploaded_csv
                df_test = load_uploaded_csv(df_test)
                success = (df_train is not None and df_test is not None)

                schema_report = validate_train_test_schema(df_train, df_test)
                if schema_report["is_compatible"]:
                    st.sidebar.success(
                        ":white_check_mark: Train/Test совместимы по схеме"
                    )
                else:
                    st.sidebar.error(":x: Схема тестовой выборки отличается")
                    st.sidebar.error(
                        ":warning: Для теста будет использована " +\
                            "часть тренировочной выборки"
                    )
                    df_test = None
        else:
            success = (df_train is not None)
    
    return df_train, df_test, data_source, has_test, summary, success


def render_eda_tab(df, summary):
    """
    Отрисовка вкладки EDA 
    
    Args:
        df: DataFrame для анализа
    """
    st.header(":mag: Exploratory Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Количество строк", summary['rows'])
    with col2:
        st.metric("Количество столбцов", summary['cols'])
    with col3:
        st.metric("Пропущенные значения", summary['missing_values'])
    with col4:
        st.metric("Дубликаты", summary['duplicates'])
    
    with st.expander(":page_facing_up: Превью данных"):
        st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander(":abacus: Статистика"):
        st.dataframe(df.describe(), use_container_width=True)
        if len(summary['categorical_cols']) > 0:
            st.dataframe(df.describe(include='object'), use_container_width=True)
    
    st.subheader(":art: Визуализация")
    
    numeric_cols = summary['numeric_cols']
    categorical_columns = summary['categorical_cols'] 

    st.markdown("**Выберите графики для отображения:**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_distribution = st.checkbox("Распределение признаков", value=True)
    with col2:
        show_scatterpots = st.checkbox("Графики рассеивания", value=True)
    with col3:
        show_correlation = st.checkbox("Матрица корреляции", value=True) 
    with col4:
        show_missing = st.checkbox("Пропущенные значения", value=False)

    if not (show_distribution or show_scatterpots or show_correlation or \
            show_missing):
        st.info(":warning: Выберите графики для отображения")
    else:
        if show_distribution:
            st.subheader("Распределение признаков")
            selected_feature = st.selectbox(
                "Выберите признак для отрисовки:", 
                df.columns
            )
            use_hue = st.checkbox("Раскрасить по категориям", value=False)
            if use_hue:
                hue_feature = st.selectbox(
                    "Выберите признак для раскрашивния:", 
                    summary['categorical_cols'],
                    key='hue_distributrion'
                )
            else:
                hue_feature = None
            fig = plot_distribution(df, selected_feature, hue_feature)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    f':warning: Для {selected_feature} невозможно'+ \
                        ' построить распределение')

        if show_scatterpots:
            st.subheader("Графики рассеивания")
            col1, col2 = st.columns(2)
            with col1:
                x = st.selectbox("X", numeric_cols)
            with col2:
                y = st.selectbox("Y", numeric_cols)
            use_hue_scat = st.checkbox(
                "Раскрасить по категориям", 
                value=False,
                key='scatterplot_hue'
            )
            if use_hue_scat:
                hue_feature = st.selectbox(
                    "Выберите признак для раскрашивния:", 
                    summary['categorical_cols'],
                    key='hue_scatterplot'
                )
            else:
                hue_feature = None
            fig = plot_scatterplot(df, x, y, hue_feature)
            st.plotly_chart(fig, use_container_width=True)
        
        if show_correlation:
            st.subheader("Матрица корреляции")
            fig = plot_correlation_matrix(df, numeric_cols)
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing values
        if show_missing:
            st.subheader("Пропущенные значения")
            missing_data = df.isnull().sum()
            if summary['missing_values'] > 0:
                fig = plot_missing_values(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(":white_check_mark: Пропущенных значений не найдено")


def render_model_training_tab(df_train, df_test):
    """
    Отрисовка вкладки обучения модели
    
    Args:
        df_train, df_test: DataFrame для обучения
    
    Returns:
        model_config (dict) с параметрами обучения или None если не обучена
    """
    st.header(":mortar_board: Обучение модели")

    st.subheader("Подготовка данных:")
    
    summary = get_data_summary(df_train)
    numeric_cols = summary['numeric_cols']
    categorical_cols = summary['categorical_cols']
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_type = st.selectbox("Тип задачи:", ["Регрессия", "Классификация"])
    
    with col2:
        if task_type == 'Регрессия':
            target_col = st.selectbox("Целевой признак:", numeric_cols)
        else:
            target_col = st.selectbox("Целевой признак:", categorical_cols)

    
    col1, col2 = st.columns(2)
    with col1:
        if df_test is None:
            test_size = st.slider("Доля тестовой выборки:", 0.1, 0.4, 0.2)
        else:
            st.success("**Тестовая выборка загружена**")
            test_size = 0.2
    with col2:
        random_state = st.slider("Random state:", 0, 100, 42)
    
    mode_na = st.radio(
        "**Обработка пропущенных значений:**",
        ["Заполнить", "Удалить", "Оставить"],
        horizontal=True
    )
    
    normalize = st.checkbox("Нормализовать данные", value=True)
    cat_coding = st.checkbox("Закодировать категориальные признаки", 
                             value=True)
    
    st.subheader("Подготовка модели:")

    if task_type == "Регрессия":
        model_choice = st.radio(
            "Выберите модель:", 
            ["LinearRegression", "RandomForestRegression"],
            horizontal=True,
            key="regression_model"
        )
    else:
        model_choice = st.radio(
            "Выберите модель:", 
            ["LogisticRegression", "RandomForestClassifier"],
            horizontal=True,
            key="classification_model"
        )
    
    if st.button(":hammer_and_wrench: Обучить модель", 
                 use_container_width=True):
        return {
            'target_col': target_col,
            'task_type': task_type,
            'test_size': test_size,
            'random_state': random_state,
            'normalize': normalize,
            'mode_na': mode_na, 
            'cat_coding': cat_coding,
            'model_choice': model_choice,
            'train_requested': True
        }
    
    return None


def show_regression_metrics(metrics):
    """Отобразить метрики регрессии"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{metrics['r2_score']:.4f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    
    fig = plot_prediction_vs_actual(metrics['y_test'], metrics['y_pred'])
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_errors(metrics['y_test'], metrics['y_pred'])
    st.plotly_chart(fig, use_container_width=True)



def show_classification_metrics(metrics):
    """Отобразить метрики классификации"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Классы", metrics['n_classes'])
    
    st.subheader("Classification Report")
    st.dataframe(metrics['report_df'], use_container_width=True)

