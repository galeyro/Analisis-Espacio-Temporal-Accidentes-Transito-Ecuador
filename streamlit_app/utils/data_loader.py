import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data(show_spinner=False)
def load_and_clean_data(file_path):
    """
    Carga y limpia el dataset de accidentes replicando la lógica exacta del notebook.
    Maneja el renombre de columnas y la limpieza de datos espaciales.
    """
    if not os.path.exists(file_path):
        st.error(f"El archivo {file_path} no existe. Por favor verifica la ruta.")
        return pd.DataFrame()
        
    # Cargar CSV con encoding adecuado para caracteres especiales
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
        
    # 1. Renombre de columnas según fuente Kaggle / original
    rename_dict = {
        "LATITUD_Y": "LATITUD", 
        "LONGITUD_X": "LONGITUD", 
        "MES_1": "MES",
        "LESIONADOS": "HERIDOS", 
        "CAUSA_PROBABLE": "CAUSA_ACCIDENTE",
        "TIPO_DE_SINIESTRO": "TIPO_ACCIDENTE", 
        "DIA_1": "DIA_SEMANA",
        "JERARQUIA_DE_LA_VIA": "JERARQUIA_VIA"
    }
    df = df.rename(columns={k:v for k,v in rename_dict.items() if k in df.columns})
    
    if "CAUSA_ACCIDENTE" not in df.columns and "CODIGO_CAUSA" in df.columns:
        df = df.rename(columns={"CODIGO_CAUSA": "CAUSA_ACCIDENTE"})
        
    # 2. Fechas a datetime
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    elif "ANIO" in df.columns and "MES" in df.columns:
        df["FECHA"] = pd.to_datetime(df["ANIO"].astype(str) + "-" + 
                                     df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
        
    # 3. Eliminar fechas inválidas
    df = df.dropna(subset=["FECHA"])
    
    # 4. Variables de tiempo derivadas
    df['AÑO'] = df['FECHA'].dt.year
    df['MES'] = df['FECHA'].dt.month
    df['DIA_SEMANA_NUM'] = df['FECHA'].dt.dayofweek
    
    # Mapear día de la semana a nombres en español
    dias = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    df['DIA_SEMANA'] = df['DIA_SEMANA_NUM'].map(dias)
    
    # 5. Filtrar solo el periodo válido (2017-2024) para evitar outliers temporales
    df = df[(df['AÑO'] >= 2017) & (df['AÑO'] <= 2024)]
    
    # 6. Hora -> entero 0..23
    if "HORA" in df.columns:
        df["HORA"] = pd.to_datetime(df["HORA"].astype(str), errors="coerce").dt.hour
        df["HORA"] = df["HORA"].fillna(12).astype(int)
        
    # 7. Severidad (Fallecidos y Heridos)
    for c in ["FALLECIDOS", "HERIDOS"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["FALLECIDOS"] = df["FALLECIDOS"].clip(lower=0).astype(int)
    df["HERIDOS"] = df["HERIDOS"].clip(lower=0).astype(int)
    
    # Gravedad derivada
    df["GRAVEDAD"] = np.where(df["FALLECIDOS"] > 0, "MORTAL",
                     np.where(df["HERIDOS"] > 0, "GRAVE", "LEVE"))
                     
    # 8. Limpieza espacial (Ecuador continental e insular)
    # LAT_MIN, LAT_MAX = -5.1, 2.0
    # LON_MIN, LON_MAX = -92.6, -75.0
    if 'LATITUD' in df.columns and 'LONGITUD' in df.columns:
        df['LATITUD'] = pd.to_numeric(df['LATITUD'], errors='coerce')
        df['LONGITUD'] = pd.to_numeric(df['LONGITUD'], errors='coerce')
        
        mask = (df["LATITUD"].between(-5.1, 2.0)) & (df["LONGITUD"].between(-92.6, -75.0))
        df = df[mask]
        
    # 9. Duplicados
    df = df.drop_duplicates()
    
    # 10. Completar categóricas faltantes
    for c in ["PROVINCIA", "CANTON", "TIPO_ACCIDENTE", "CAUSA_ACCIDENTE", "ZONA"]:
        if c not in df.columns: df[c] = "N/D"
        else: df[c] = df[c].fillna("N/D")
        
    # Normalizar provincia a mayúsculas para evitar dobles registros
    df['PROVINCIA'] = df['PROVINCIA'].astype(str).str.upper().str.strip()
        
    df = df.sort_values("FECHA").reset_index(drop=True)
    
    return df
