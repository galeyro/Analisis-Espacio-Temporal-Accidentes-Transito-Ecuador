import streamlit as st
import pandas as pd
import sys
import os

# Añadir config al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.plots import plot_forecast_comparison

st.set_page_config(page_title="Pronóstico Temporal", page_icon="📈", layout="wide")
st.title("📈 Pronóstico Temporal de Accidentes")

# Función para cargar resultados
@st.cache_data
def load_forecast_data():
    results, metrics = pd.DataFrame(), pd.DataFrame()
    if os.path.exists(config.FORECAST_RESULTS_FILE):
        results = pd.read_csv(config.FORECAST_RESULTS_FILE)
    if os.path.exists(config.FORECAST_METRICS_FILE):
        metrics = pd.read_csv(config.FORECAST_METRICS_FILE)
    return results, metrics

with st.spinner("Cargando predicciones precomputadas..."):
    df_results, df_metrics = load_forecast_data()

if df_results.empty or df_metrics.empty:
    st.warning("Los resultados del pronóstico no están disponibles. Asegúrate de ejecutar el script `scripts/precompute_forecasts.py` primero.")
    st.stop()

st.markdown("""
Esta sección muestra las predicciones generadas por múltiples modelos de Machine Learning (SARIMA, Prophet, LSTM, XGBoost, Random Forest) y un modelo Ensemble.
Los resultados mostrados corresponden a la fase de validación (prediciendo los últimos 12 meses históricos).
""")

# Selector de modelos
st.sidebar.header("Opciones de Visualización")
modelos_disponibles = [col for col in df_results.columns if col not in ['FECHA_MES', 'Real', 'index']]
selected_models = st.sidebar.multiselect(
    "Selecciona los modelos a comparar", 
    options=modelos_disponibles,
    default=modelos_disponibles
)

# Filtrar df_results
cols_to_plot = []
if 'FECHA_MES' in df_results.columns:
    cols_to_plot.append('FECHA_MES')
elif 'index' in df_results.columns:
    cols_to_plot.append('index')
    
cols_to_plot.append('Real')
cols_to_plot.extend(selected_models)

df_plot = df_results[cols_to_plot]

# Mostrar gráfico
st.plotly_chart(plot_forecast_comparison(df_plot), use_container_width=True)

st.markdown("---")
st.markdown("### 📊 Desempeño de los Modelos (Métricas de Error)")
st.markdown("""
- **RMSE** (Root Mean Squared Error): Penaliza errores grandes (menor es mejor).
- **MAE** (Mean Absolute Error): Error promedio absoluto en número de accidentes (menor es mejor).
- **MAPE** (Mean Absolute Percentage Error): Error en porcentaje relativo al valor real (menor es mejor).
- **R²** (R-squared): Varianza explicada por el modelo (cercano a 1 es mejor).
""")

# Mostrar tabla de métricas estilizada
if not df_metrics.empty:
    # Resaltar los mejores valores
    st.dataframe(
        df_metrics.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='#065f46')
                  .format({
                      'RMSE': '{:.2f}',
                      'MAE': '{:.2f}',
                      'MAPE': '{:.2f}%'
                  }),
        use_container_width=True
    )
