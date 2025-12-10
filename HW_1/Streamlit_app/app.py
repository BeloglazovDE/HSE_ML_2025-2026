"""
Основной файл запуска приложения

Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import (
    load_example_data, 
    load_uploaded_csv
)
from model_training import (
    prepare_data, 
    model_training_func,
    make_prediction,
    get_model_weights
)
from visualizations import (
    plot_coefficients,
    plot_feature_importances
)
from ui_components import (
    render_header,
    render_data_loading_sidebar,
    render_eda_tab,
    render_model_training_tab,
    show_regression_metrics,
    show_classification_metrics,
)


st.set_page_config(
    page_title="ML Service",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_header()

if 'df_train' not in st.session_state:
    st.session_state.df_train = None
if 'df_test' not in st.session_state:
    st.session_state.df_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None


(
    df_train, 
    df_test, 
    data_source, 
    has_test, 
    summary,
    success 
) = render_data_loading_sidebar()

if success:
    st.session_state.df_train = df_train
    st.session_state.df_test = df_test
    
    tab1, tab2, tab3 = st.tabs([
        ":mag: EDA", 
        ":mortar_board: Обучение модели", 
        ":bar_chart: Веса модели"
    ])
    
    with tab1:
        render_eda_tab(df_train, summary)
    
    
    with tab2:
        model_config = render_model_training_tab(df_train, df_test)
        
        if model_config and model_config.get('train_requested'):
            try:
                X_train, X_test, y_train, y_test, transformer, feature_names =\
                    prepare_data(
                        df_train=df_train,
                        df_test=df_test,
                        model_config=model_config
                    )
                
                model, metrics = model_training_func(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_config=model_config
                )

                st.success(":white_check_mark: Модель успешно обучена!")
                
                st.session_state.model = model
                st.session_state.scaler = transformer
                st.session_state.model_metrics = metrics
                st.session_state.feature_names = feature_names
                st.session_state.task_type = model_config['task_type']
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Display metrics
                if model_config['task_type'] == "Регрессия":
                    show_regression_metrics(metrics)
                else:
                    show_classification_metrics(metrics)
            
            except Exception as e:
                st.error(f":x: Ошибка при обучении: {e}")
    
    with tab3:
        st.header(":bar_chart: Визуализация весов модели")
        
        if st.session_state.model is None:
            st.warning(":warning: Сначала обучите модель на вкладке "+ \
                       "'Обучение модели'")
        else:
            model = st.session_state.model
            feature_names = st.session_state.feature_names
            
            weights, has_coef, has_importances = get_model_weights(model)
            
            if has_coef:
                st.subheader(
                    f"Коэффициенты модели {model_config['model_choice']}"
                )
                coef = weights
                
                if len(coef.shape) > 1 and coef.shape[0] > 1:
                    selected_class = st.selectbox("Выберите класс:", 
                                                  range(coef.shape[0]))
                    coef = coef[selected_class]
                
                fig, coef_df = plot_coefficients(feature_names, coef)
                
                st.dataframe(coef_df, use_container_width=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif has_importances:
                st.subheader(
                    f"Важность признаков {model_config['model_choice']}"
                )
                importances = weights
                
                fig, importance_df = plot_feature_importances(
                    feature_names, 
                    importances
                )

                st.dataframe(importance_df, use_container_width=True)
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning(
                    ":warning: Выбранная модель не имеет визуализируемых весов"
                )

else:
    st.warning(
        ":file_folder: Загрузите данные слева в боковой панели, чтобы начать!"
    )
