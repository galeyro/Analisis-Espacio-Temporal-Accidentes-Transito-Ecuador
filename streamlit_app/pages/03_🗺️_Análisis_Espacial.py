import streamlit as st
import sys
import os
from streamlit_folium import st_folium

# Añadir config al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.data_loader import load_and_clean_data
from utils.dbscan_model import run_dbscan, create_folium_map

st.set_page_config(page_title="Análisis Espacial", page_icon="🗺️", layout="wide")
st.title("🗺️ Análisis Espacial (Hotspots DBSCAN)")

with st.spinner("Cargando datos..."):
    df = load_and_clean_data(config.DATA_FILE)

if df.empty:
    st.warning("No hay datos para mostrar.")
    st.stop()

# Panel lateral para controles DBSCAN
st.sidebar.header("Parámetros DBSCAN")
st.sidebar.markdown("""
El algoritmo DBSCAN agrupa puntos cercanos basados en una distancia máxima (**Epsilon**) 
y un número mínimo de puntos (**Min Samples**).
""")

eps_km = st.sidebar.slider("Epsilon (km) - Radio de búsqueda", 
                           min_value=0.1, max_value=5.0, value=config.DEFAULT_EPS, step=0.1)
min_samples = st.sidebar.slider("Min Samples - Puntos mínimos", 
                                min_value=2, max_value=20, value=config.DEFAULT_MIN_SAMPLES, step=1)
sample_size = st.sidebar.selectbox("Muestra de datos (Mejora el rendimiento)", 
                                   options=[5000, 10000, 15000, 30000], index=2)

st.info(f"Mostrando un análisis interactivo sobre {sample_size} registros para optimizar el rendimiento.")

# Ejecutar DBSCAN
with st.spinner("Ejecutando clustering espacial (puede tomar unos segundos)..."):
    df_clustered, metrics = run_dbscan(df, eps_km=eps_km, min_samples=min_samples, sample_size=sample_size)

# Mostrar métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters Encontrados", metrics['n_clusters'])
col2.metric("Puntos de Ruido", f"{metrics['n_noise']} ({(metrics['n_noise']/metrics['total_points'])*100:.1f}%)")

if metrics['silhouette']:
    col3.metric("Silhouette Score", f"{metrics['silhouette']:.3f}", help="Cercano a 1 es mejor. Evalúa separación y cohesión.")
if metrics['davies_bouldin']:
    col4.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}", help="Menor es mejor. Evalúa dispersión.")

st.markdown("---")

# Renderizar Mapa
st.markdown("### Mapa de Calor y Clusters")
m = create_folium_map(df_clustered)
import streamlit.components.v1 as components
components.html(m._repr_html_(), height=600)

# Resumen de Clusters
if metrics['n_clusters'] > 0:
    st.markdown("### Resumen por Cluster")
    # Agrupar datos de clusters (excluyendo ruido -1)
    clusters_only = df_clustered[df_clustered['Cluster'] != -1]
    
    if not clusters_only.empty:
        # Calcular gravedad si existe
        agg_dict = {'LATITUD': 'count'}
        if 'FALLECIDOS' in clusters_only.columns:
            agg_dict['FALLECIDOS'] = 'sum'
        if 'HERIDOS' in clusters_only.columns:
            agg_dict['HERIDOS'] = 'sum'
            
        resumen = clusters_only.groupby('Cluster').agg(agg_dict).reset_index()
        resumen = resumen.rename(columns={'LATITUD': 'Total Accidentes'})
        resumen = resumen.sort_values(by='Total Accidentes', ascending=False)
        
        st.dataframe(resumen, use_container_width=True)
