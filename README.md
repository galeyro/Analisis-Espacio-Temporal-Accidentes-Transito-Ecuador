# 🚗 Análisis Espacio-Temporal de Accidentes de Tránsito en Ecuador

Este proyecto final de Inteligencia Artificial II se centra en el análisis detallado, tanto espacial como temporal, de los accidentes de tránsito ocurridos en Ecuador (2017 - 2024). El objetivo principal es identificar patrones de siniestralidad, puntos calientes (hotspots) y realizar pronósticos para ayudar en la prevención y toma de decisiones.

El proyecto está dividido en dos partes principales: un **Notebook de Análisis en Profundidad** y un **Dashboard Interactivo en Streamlit**.

---

## 📓 1. Notebook Principal (`Proyecto_Final_IA2_Accidentes_Ecuador_v2.ipynb`)

Este Jupyter Notebook contiene todo el pipeline de Data Science e Inteligencia Artificial, detallando paso a paso:

- **Procesamiento y Limpieza de Datos:** Tratamiento de la base de datos original (`BDD_ENE_17_ABR_24.csv`).
- **Análisis Exploratorio de Datos (EDA):** Visualizaciones estadísticas sobre las causas principales, severidad, rangos de horarios y lugares más frecuentes de los accidentes.
- **Análisis Espacial:** Aplicación del algoritmo de clustering **DBSCAN** para identificar agrupaciones geográficas o *'Hotspots'* de siniestralidad.
- **Modelos de Pronóstico Temporal:** Entrenamiento y evaluación comparativa de algoritmos de Machine Learning y Deep Learning para predecir la cantidad de accidentes futuros. Se implementaron los siguientes modelos:
  - Modelos Estadísticos y Tradicionales: **SARIMA**, **Prophet**
  - Machine Learning: **XGBoost**
  - Deep Learning: Redes Neuronales Recurrentes **LSTM**

---

## 🖥️ 2. Dashboard Interactivo (Extra: `streamlit_app`)

Como un valor agregado al proyecto, se desarrolló una aplicación web interactiva utilizando **Streamlit**. Esta herramienta permite consumir los resultados del análisis y los modelos de manera visual y accesible para cualquier usuario.

La aplicación cuenta con una barra de navegación lateral organizada en los siguientes módulos:
- **🎯 KPIs (`01_🎯_KPIs.py`):** Resumen ejecutivo con las métricas más importantes (Total de accidentes, número de fallecidos y heridos).
- **📊 Análisis Exploratorio (`02_📊_Análisis_Exploratorio.py`):** Gráficos dinámicos para explorar las estadísticas generales.
- **🗺️ Análisis Espacial (`03_🗺️_Análisis_Espacial.py`):** Mapas interactivos que muestran las zonas de alto riesgo identificadas.
- **📈 Pronóstico Temporal (`04_📈_Pronóstico_Temporal.py`):** Visualización interactiva de las predicciones futuras generadas por los distintos modelos predictivos.

### 🚀 ¿Cómo ejecutar la aplicación localmente?

1. Asegúrate de tener todas las dependencias necesarias instaladas. Puedes hacerlo ejecutando:
   ```bash
   pip install -r streamlit_app/requirements.txt
   ```

2. Para levantar la aplicación, debes ejecutar el archivo principal de Streamlit. Desde la raíz de tu proyecto, corre el siguiente comando en la terminal:
   ```bash
   streamlit run streamlit_app/01_🎯_KPIs.py
   ```
   
   *(Nota: Si ya ingresaste a la carpeta con `cd streamlit_app`, el comando sería simplemente `streamlit run 01_🎯_KPIs.py`)*

---
**Desarrollado para:** Proyecto Final de IA 2

### 👥 Integrantes del Equipo
- Mateo Arguello
- Joaquin Chacon
- Andres Jimenez
- Mathew Baquero
- Galo Guevara
- Joel Ibarra

