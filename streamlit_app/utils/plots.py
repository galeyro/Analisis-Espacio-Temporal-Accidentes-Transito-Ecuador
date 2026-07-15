import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Añadir config al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Configuración base para gráficos oscuros
template = "plotly_dark"
color_discrete_sequence = [config.COLOR_ACCENT, config.COLOR_INFO, "#10B981", "#8B5CF6", "#EC4899"]

def plot_accidents_over_time(df):
    """Serie temporal de accidentes por mes"""
    if 'FECHA' not in df.columns: return go.Figure()
    
    # Agrupar por mes
    df_temp = df.copy()
    df_temp['Mes_Año'] = df_temp['FECHA'].dt.to_period('M').dt.to_timestamp()
    ts = df_temp.groupby('Mes_Año').size().reset_index(name='Accidentes')
    
    fig = px.line(ts, x='Mes_Año', y='Accidentes', 
                  title='Tendencia Mensual de Accidentes de Tránsito',
                  template=template,
                  color_discrete_sequence=[config.COLOR_INFO])
    fig.update_traces(line=dict(width=3))
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Número de Accidentes")
    return fig

def plot_accidents_by_hour(df):
    """Distribución de accidentes por hora del día"""
    if 'HORA' not in df.columns: return go.Figure()
    
    df_temp = df.copy()
    
    # Si la hora ya es numérica (entera o float), la usamos directamente
    if pd.api.types.is_numeric_dtype(df_temp['HORA']):
        df_temp['Solo_Hora'] = df_temp['HORA'].fillna(12).astype(int)
    else:
        # Intentamos extraerla si es string
        try:
            df_temp['Solo_Hora'] = pd.to_datetime(df_temp['HORA'], format='%H:%M:%S', errors='coerce').dt.hour
            if df_temp['Solo_Hora'].isna().all():
                df_temp['Solo_Hora'] = pd.to_datetime(df_temp['HORA'], format='%H:%M', errors='coerce').dt.hour
        except:
            return go.Figure()
        
    counts = df_temp['Solo_Hora'].value_counts().sort_index().reset_index()
    counts.columns = ['Hora', 'Accidentes']
    
    fig = px.bar(counts, x='Hora', y='Accidentes',
                 title='Accidentes por Hora del Día',
                 template=template,
                 color_discrete_sequence=[config.COLOR_ACCENT])
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    return fig

def plot_accidents_by_weekday(df):
    """Distribución por día de la semana"""
    if 'DIA_SEMANA' not in df.columns: return go.Figure()
    
    dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    counts = df['DIA_SEMANA'].value_counts().reindex(dias_orden).reset_index()
    counts.columns = ['Dia', 'Accidentes']
    
    fig = px.bar(counts, x='Dia', y='Accidentes',
                 title='Accidentes por Día de la Semana',
                 template=template,
                 color='Accidentes', color_continuous_scale='Oranges')
    return fig

def plot_top_provinces(df, n=10):
    """Top N provincias con más accidentes"""
    if 'PROVINCIA' not in df.columns: return go.Figure()
    
    counts = df['PROVINCIA'].value_counts().head(n).reset_index()
    counts.columns = ['Provincia', 'Accidentes']
    
    fig = px.bar(counts, x='Accidentes', y='Provincia', orientation='h',
                 title=f'Top {n} Provincias con más Accidentes',
                 template=template,
                 color='Accidentes', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_severity(df):
    """Gráfico de dona para gravedad"""
    col = None
    if 'GRAVEDAD' in df.columns: col = 'GRAVEDAD'
    elif 'ZONA' in df.columns: col = 'ZONA' # Si no hay gravedad, mostrar zona como alternativo
    else: return go.Figure()
        
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'Cantidad']
    
    fig = px.pie(counts, values='Cantidad', names=col, hole=0.5,
                 title=f'Distribución por {col}',
                 template=template,
                 color_discrete_sequence=color_discrete_sequence)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_forecast_comparison(df_results):
    """Gráfico de líneas comparando modelos con datos reales"""
    if df_results.empty: return go.Figure()
    
    # Asumimos que df_results tiene 'FECHA_MES', 'Real', 'SARIMA', etc.
    if 'FECHA_MES' in df_results.columns:
        df_melt = df_results.melt(id_vars=['FECHA_MES'], var_name='Modelo', value_name='Accidentes')
        x_col = 'FECHA_MES'
    else:
        df_melt = df_results.reset_index().melt(id_vars=['index'], var_name='Modelo', value_name='Accidentes')
        x_col = 'index'
        
    fig = px.line(df_melt, x=x_col, y='Accidentes', color='Modelo',
                  title='Comparación de Modelos de Pronóstico',
                  template=template,
                  color_discrete_sequence=['white'] + color_discrete_sequence) # Real en blanco
                  
    # Hacer la línea real más gruesa
    for trace in fig.data:
        if trace.name == 'Real':
            trace.line.width = 4
        else:
            trace.line.dash = 'dot'
            trace.line.width = 2
            
    fig.update_layout(hovermode="x unified")
    return fig
