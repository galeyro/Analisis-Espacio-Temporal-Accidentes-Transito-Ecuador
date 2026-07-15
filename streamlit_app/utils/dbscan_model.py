import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
from folium.plugins import HeatMap
import streamlit as st
import config

@st.cache_data
def run_dbscan(df, eps_km=config.DEFAULT_EPS, min_samples=config.DEFAULT_MIN_SAMPLES, sample_size=15000):
    """
    Ejecuta DBSCAN usando distancia Haversine.
    Toma un sample aleatorio si el dataset es muy grande para no bloquear la app.
    """
    df_spatial = df.dropna(subset=['LATITUD', 'LONGITUD']).copy()
    
    if len(df_spatial) > sample_size:
        df_spatial = df_spatial.sample(n=sample_size, random_state=42)
        
    coords = df_spatial[['LATITUD', 'LONGITUD']].values
    # Convertir a radianes para Haversine
    coords_rad = np.radians(coords)
    
    # eps en radianes (Radio de la tierra = 6371.0088 km)
    earth_radius_km = 6371.0088
    eps_rad = eps_km / earth_radius_km
    
    # Configurar y entrenar DBSCAN
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    df_spatial['Cluster'] = db.fit_predict(coords_rad)
    
    # Calcular métricas (solo si hay más de 1 cluster excluyendo el ruido)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'total_points': len(df_spatial),
        'silhouette': None,
        'davies_bouldin': None
    }
    
    if 1 < n_clusters < len(df_spatial) - 1:
        # Extraer puntos que no son ruido para las métricas
        core_samples = df_spatial[df_spatial['Cluster'] != -1]
        if len(core_samples) > 0 and len(core_samples['Cluster'].unique()) > 1:
            metrics['silhouette'] = silhouette_score(
                np.radians(core_samples[['LATITUD', 'LONGITUD']].values), 
                core_samples['Cluster'], metric='haversine'
            )
            metrics['davies_bouldin'] = davies_bouldin_score(
                core_samples[['LATITUD', 'LONGITUD']].values, 
                core_samples['Cluster']
            )
            
    return df_spatial, metrics

def create_folium_map(df_clustered):
    """Crea el mapa de calor y marcadores de clusters"""
    # Centrar el mapa en Ecuador
    m = folium.Map(location=[-1.8312, -78.1834], zoom_start=6, tiles='CartoDB dark_matter')
    
    # Capa de HeatMap para los puntos de ruido
    ruido = df_clustered[df_clustered['Cluster'] == -1]
    if not ruido.empty:
        heat_data = [[row['LATITUD'], row['LONGITUD']] for index, row in ruido.iterrows()]
        HeatMap(heat_data, radius=10, blur=15, gradient={0.4: 'blue', 0.6: 'cyan', 0.8: 'lime', 1: 'yellow'}).add_to(m)
        
    # Capa de marcadores para los clusters identificados
    clusters = df_clustered[df_clustered['Cluster'] != -1]
    if not clusters.empty:
        # Colores para diferentes clusters
        colors = ['red', 'orange', 'green', 'purple', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
        
        for cluster_id in clusters['Cluster'].unique():
            cluster_data = clusters[clusters['Cluster'] == cluster_id]
            color = colors[cluster_id % len(colors)]
            
            # Dibujar un círculo en el centroide del cluster
            centro_lat = cluster_data['LATITUD'].mean()
            centro_lon = cluster_data['LONGITUD'].mean()
            
            folium.CircleMarker(
                location=[centro_lat, centro_lon],
                radius=10,
                popup=f"Cluster {cluster_id}<br>Accidentes: {len(cluster_data)}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
            
            # (Opcional) Dibujar puntos individuales si son pocos
            if len(cluster_data) < 1000:
                for idx, row in cluster_data.iterrows():
                    folium.CircleMarker(
                        location=[row['LATITUD'], row['LONGITUD']],
                        radius=2,
                        color=color,
                        weight=1,
                        opacity=0.5
                    ).add_to(m)
                    
    return m
