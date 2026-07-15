import streamlit as st
import sys
import os

# Añadir config al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.data_loader import load_and_clean_data
from utils.plots import (plot_accidents_over_time, plot_accidents_by_hour,
                         plot_accidents_by_weekday, plot_top_provinces,
                         plot_severity)

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Análisis Exploratorio de Datos (EDA)")

with st.spinner("Cargando datos..."):
    df = load_and_clean_data(config.DATA_FILE)

if df.empty:
    st.warning("No hay datos para mostrar.")
    st.stop()

# Filtros
st.sidebar.header("Filtros")
years = sorted(df['AÑO'].dropna().unique().tolist())
selected_years = st.sidebar.multiselect("Año(s)", years, default=years)

if 'PROVINCIA' in df.columns:
    provinces = sorted(df['PROVINCIA'].dropna().unique().tolist())
    selected_provinces = st.sidebar.multiselect("Provincia(s)", provinces)
else:
    selected_provinces = []

# Aplicar filtros
df_filtered = df.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered['AÑO'].isin(selected_years)]
if selected_provinces:
    df_filtered = df_filtered[df_filtered['PROVINCIA'].isin(selected_provinces)]

st.markdown(f"**Registros mostrados:** {len(df_filtered):,}")

# Primera fila
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_accidents_over_time(df_filtered), use_container_width=True)
with col2:
    st.plotly_chart(plot_accidents_by_hour(df_filtered), use_container_width=True)

# Segunda fila
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(plot_accidents_by_weekday(df_filtered), use_container_width=True)
with col4:
    st.plotly_chart(plot_top_provinces(df_filtered), use_container_width=True)

# Tercera fila
col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(plot_severity(df_filtered), use_container_width=True)
with col6:
    pass # Espacio para otra métrica
