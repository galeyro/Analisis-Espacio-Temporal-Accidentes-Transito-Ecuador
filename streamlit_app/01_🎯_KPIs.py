import streamlit as st
import sys
import os

# Añadir config al path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config
from utils.data_loader import load_and_clean_data

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Accidentes Ecuador",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado (CSS)
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {config.COLOR_BG};
    }}
    .metric-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {config.COLOR_ACCENT};
        margin: 0;
    }}
    .metric-label {{
        font-size: 1rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🚗 Análisis de Accidentes de Tránsito en Ecuador")
    st.markdown("### Resumen Ejecutivo (2017 - 2024)")
    
    with st.spinner("Cargando datos..."):
        df = load_and_clean_data(config.DATA_FILE)
        
    if df.empty:
        st.warning("No hay datos para mostrar. Revisa el archivo CSV.")
        return
        
    # KPIs
    total_accidentes = len(df)
    total_fallecidos = int(df['FALLECIDOS'].sum()) if 'FALLECIDOS' in df.columns else 0
    total_heridos = int(df['HERIDOS'].sum()) if 'HERIDOS' in df.columns else 0
    
    # Renderizar KPIs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total Accidentes</p>
                <p class="metric-value">{total_accidentes:,.0f}</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total Fallecidos</p>
                <p style="color: #EF4444;" class="metric-value">{total_fallecidos:,.0f}</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total Heridos</p>
                <p style="color: #F59E0B;" class="metric-value">{total_heridos:,.0f}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 🎯 Sobre este Dashboard
    Esta aplicación interactiva es el resultado del análisis espacio-temporal de accidentes de tránsito en Ecuador.
    
    Utiliza el menú lateral para navegar entre:
    - **📊 Análisis Exploratorio**: Gráficos interactivos para entender las causas, horarios y lugares más frecuentes.
    - **🗺️ Análisis Espacial (DBSCAN)**: Identificación de 'Hotspots' o puntos calientes de siniestralidad.
    - **📈 Pronóstico Temporal**: Comparación de modelos de Machine Learning (SARIMA, LSTM, Prophet, XGBoost) para predecir futuros accidentes.
    """)

if __name__ == "__main__":
    main()
