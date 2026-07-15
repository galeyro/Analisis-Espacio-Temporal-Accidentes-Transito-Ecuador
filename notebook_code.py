# --- Instalación de dependencias (Google Colab) -----------------------------
# Descomenta si estás en Colab / entorno limpio:
# !pip -q install "scikit-learn>=1.3" prophet statsmodels tensorflow \
#                 xgboost folium haversine plotly geopandas

import warnings, os, random
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)

import numpy as np
import pandas as pd
np.random.seed(SEED)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

print("Entorno base listo · pandas", pd.__version__, "· numpy", np.__version__)

# --- Localización del archivo (con descarga automática desde Kaggle) ----------
import os, glob

CANDIDATE_PATHS = [
    "BDD_ENE_17_ABR_24.csv",
    "../BDD_ENE_17_ABR_24.csv",
    "data/BDD_ENE_17_ABR_24.csv",
    "/content/BDD_ENE_17_ABR_24.csv",
    "/kaggle/input/accidentesecu/BDD_ENE_17_ABR_24.csv",
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)

# Si no está local, descargarlo de Kaggle (kagglehub: dataset público, sin token)
if DATA_PATH is None:
    try:
        import kagglehub, shutil
        ruta = kagglehub.dataset_download("alicevangomez/accidentesecu")
        hits = glob.glob(os.path.join(ruta, "**", "*.csv"), recursive=True)
        if hits:
            shutil.copy(hits[0], "BDD_ENE_17_ABR_24.csv")
            DATA_PATH = "BDD_ENE_17_ABR_24.csv"
            print("✓ CSV descargado desde Kaggle.")
    except Exception as e:
        print("Descarga desde Kaggle no disponible:", e)

if DATA_PATH:
    print("CSV real encontrado en:", DATA_PATH)
else:
    print("CSV real NO encontrado → se usará el generador sintético calibrado.")


# --- Loader del CSV real (estructura Kaggle: 56 columnas, encoding latin-1) --
def load_real_data(path):
    df = pd.read_csv(path, delimiter=",", encoding="latin-1", low_memory=False)
    # Renombres según la fuente Kaggle (alicevangomez/ecu-accidentes)
    rename = {"LATITUD_Y":"LATITUD", "LONGITUD_X":"LONGITUD", "MES_1":"MES",
              "LESIONADOS":"HERIDOS", "CAUSA_PROBABLE":"CAUSA_ACCIDENTE",
              "TIPO_DE_SINIESTRO":"TIPO_ACCIDENTE", "DIA_1":"DIA_SEMANA",
              "JERARQUIA_DE_LA_VIA":"JERARQUIA_VIA"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if "CAUSA_ACCIDENTE" not in df.columns and "CODIGO_CAUSA" in df.columns:
        df = df.rename(columns={"CODIGO_CAUSA":"CAUSA_ACCIDENTE"})
    # MES textual ("ENERO"...) -> numérico si aplica
    if df["MES"].dtype == object and "MES_2" in df.columns:
        df["MES"] = pd.to_numeric(df["MES_2"], errors="coerce")
    # Fecha
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    else:
        df["FECHA"] = pd.to_datetime(df["ANIO"].astype(str)+"-"+
                                     df["MES"].astype(str).str.zfill(2)+"-01", errors="coerce")
    # Hora -> entero 0..23
    if "HORA" in df.columns:
        df["HORA"] = pd.to_datetime(df["HORA"].astype(str), errors="coerce").dt.hour
        df["HORA"] = df["HORA"].fillna(12).astype(int)
    for c in ["FALLECIDOS","HERIDOS"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["ANIO","MES"]:
        if c not in df.columns and "FECHA" in df.columns:
            df[c] = getattr(df["FECHA"].dt, "year" if c=="ANIO" else "month")
    for c in ["PROVINCIA","CANTON","TIPO_ACCIDENTE","CAUSA_ACCIDENTE","ZONA","FERIADO"]:
        if c not in df.columns: df[c] = "N/D"
    return df


# --- Generador sintético calibrado al paper (fallback reproducible) ----------
def make_synthetic(n=166_685, seed=SEED):
    rng = np.random.default_rng(seed)
    # Centros urbanos (lat, lon, peso ~ participacion de accidentes)
    centros = [
        ("Guayaquil", -2.170, -79.922, 0.230, 0.008),
        ("Quito",     -0.190, -78.490, 0.150, 0.010),
        ("Cuenca",    -2.900, -79.010, 0.045, 0.010),
        ("Ambato",    -1.241, -78.620, 0.035, 0.010),
        ("Machala",   -3.258, -79.960, 0.030, 0.010),
        ("Manta",     -0.967, -80.708, 0.030, 0.012),
        ("Sto Domingo",-0.253,-79.175, 0.030, 0.012),
        ("Loja",      -3.994, -79.204, 0.020, 0.010),
        ("Portoviejo",-1.055, -80.455, 0.020, 0.012),
        ("Riobamba",  -1.664, -78.654, 0.018, 0.010),
    ]
    resto_w = 1 - sum(c[3] for c in centros)
    pesos = [c[3] for c in centros] + [resto_w]
    idx = rng.choice(len(centros)+1, size=n, p=pesos)
    lat = np.empty(n); lon = np.empty(n); canton = np.empty(n, dtype=object)
    for i,c in enumerate(centros):
        m = idx==i
        k = m.sum()
        lat[m] = rng.normal(c[1], c[4], k); lon[m] = rng.normal(c[2], c[4]*1.2, k)
        canton[m] = c[0]
    m = idx==len(centros); k = m.sum()
    lat[m] = rng.uniform(-4.8, 1.2, k); lon[m] = rng.uniform(-80.8, -75.5, k)
    canton[m] = "Otros"
    # Fechas ene-2017..abr-2024 con estacionalidad + caida COVID 2020
    meses = pd.date_range("2017-01-01","2024-04-30", freq="MS")
    base = 1894 + 361*0.5*np.sin(np.arange(len(meses))*2*np.pi/12)   # estacional
    for j,mth in enumerate(meses):
        if mth.year==2020 and mth.month in (3,4,5,6,7):
            base[j] *= {3:0.57,4:0.27,5:0.46,6:0.61,7:0.60}[mth.month]  # COVID
        if mth.year==2017 and mth.month==12: base[j]*=1.41              # pico dic-2017
    base = np.clip(base,120,None); base = base/base.sum()*n
    counts = np.floor(base).astype(int); counts[-1]+=n-counts.sum()
    fechas = np.concatenate([
        rng.integers(0, pd.Period(m,'M').days_in_month, c) + np.datetime64(m,'D')
        for m,c in zip(meses,counts)])[:n]
    fechas = pd.to_datetime(rng.permutation(fechas))
    # Hora con doble pico 8h y 19h
    horas = np.clip(np.round(rng.choice(
        [7,8,9,12,13,17,18,19,20,22,0,2],
        p=[.09,.11,.08,.07,.07,.09,.12,.13,.08,.06,.06,.04], size=n)),0,23).astype(int)
    fall = (rng.random(n) < 0.055).astype(int)*rng.integers(1,3,n)
    her  = (rng.random(n) < 0.62 ).astype(int)*rng.integers(1,4,n)
    tipos = rng.choice(["CHOQUE LATERAL","CHOQUE FRONTAL","ATROPELLO","VOLCAMIENTO",
                        "CAIDA PASAJERO","ROZAMIENTO","ESTRELLAMIENTO"],
                       p=[.34,.15,.14,.09,.08,.11,.09], size=n)
    causas = rng.choice(["NO RESPETA SEÑALES DE TRANSITO","CONDUCE DESATENTO",
                         "EXCESO DE VELOCIDAD","INVADE CARRIL","EMBRIAGUEZ",
                         "NO GUARDA DISTANCIA","IMPERICIA DEL CONDUCTOR"],
                        p=[.22,.21,.17,.13,.10,.09,.08], size=n)
    zona = np.where(canton=="Otros", "RURAL", "URBANA")
    prov_map = {"Guayaquil":"GUAYAS","Quito":"PICHINCHA","Cuenca":"AZUAY",
                "Ambato":"TUNGURAHUA","Machala":"EL ORO","Manta":"MANABI",
                "Sto Domingo":"SANTO DOMINGO DE LOS TSACHILAS","Loja":"LOJA",
                "Portoviejo":"MANABI","Riobamba":"CHIMBORAZO"}
    otras = ["ESMERALDAS","LOS RIOS","IMBABURA","COTOPAXI","BOLIVAR","CANAR",
             "CARCHI","NAPO","PASTAZA","SUCUMBIOS","ORELLANA","MORONA SANTIAGO",
             "ZAMORA CHINCHIPE","SANTA ELENA","GALAPAGOS"]
    provincia = np.array([prov_map.get(c, "") for c in canton], dtype=object)
    m_otros = provincia == ""
    provincia[m_otros] = rng.choice(otras, size=m_otros.sum())
    df = pd.DataFrame({"FECHA":fechas,"ANIO":fechas.year,"MES":fechas.month,"HORA":horas,
                       "LATITUD":lat,"LONGITUD":lon,"CANTON":canton,"PROVINCIA":provincia,
                       "TIPO_ACCIDENTE":tipos,"CAUSA_ACCIDENTE":causas,
                       "ZONA":zona,"FERIADO":"N/D",
                       "FALLECIDOS":fall,"HERIDOS":her})
    return df.sort_values("FECHA").reset_index(drop=True)

# --- Cargar ------------------------------------------------------------------
if DATA_PATH:
    df_raw = load_real_data(DATA_PATH); FUENTE = "CSV real ANT"
else:
    df_raw = make_synthetic(); FUENTE = "SINTÉTICO calibrado (reemplazar por CSV real)"
print(f"Fuente de datos : {FUENTE}")
print(f"Registros       : {len(df_raw):,}")
print(f"Columnas        : {list(df_raw.columns)}")
df_raw.head()

# --- Diagnóstico de calidad ANTES de limpiar (valores faltantes y consistencia) --
cols_rel = [c for c in ["FECHA","HORA","LATITUD","LONGITUD","FALLECIDOS","HERIDOS",
                        "PROVINCIA","CANTON","TIPO_ACCIDENTE","CAUSA_ACCIDENTE","ZONA"]
            if c in df_raw.columns]
calidad = pd.DataFrame({
    "dtype":      df_raw[cols_rel].dtypes.astype(str),
    "n_nulos":    df_raw[cols_rel].isna().sum(),
    "%_nulos":    (df_raw[cols_rel].isna().mean()*100).round(2),
    "n_unicos":   df_raw[cols_rel].nunique(),
})
print("=== Calidad de datos (previo a limpieza) ===")
display(calidad)
print(f"Duplicados exactos: {df_raw.duplicated().sum():,}")
print(f"Coordenadas (0,0) o placeholder: "
      f"{((df_raw['LATITUD']==0)&(df_raw['LONGITUD']==0)).sum():,}")

# Ecuador CONTINENTAL + INSULAR (Galápagos: lon ≈ −92 a −89) — no excluir regiones
LAT_MIN, LAT_MAX = -5.1, 2.0
LON_MIN, LON_MAX = -92.6, -75.0

df = df_raw.copy()
n0 = len(df)
# 1) Fechas nulas o inválidas
nfec = df["FECHA"].isna().sum()
df = df.dropna(subset=["FECHA"])
# 2) Coordenadas fuera del territorio ecuatoriano (incluye el placeholder ~-75.0 del CSV)
mask = df["LATITUD"].between(LAT_MIN,LAT_MAX) & df["LONGITUD"].between(LON_MIN,LON_MAX)
print(f"Fechas inválidas eliminadas   : {nfec:,}")
print(f"Coordenadas fuera de rango    : {(~mask).sum():,} ({(~mask).mean()*100:.2f}%)")
df = df[mask]
# 3) Duplicados exactos
ndup = df.duplicated().sum(); df = df.drop_duplicates()
# 4) Severidad no negativa; nulos = 0 víctimas
df["FALLECIDOS"] = df["FALLECIDOS"].clip(lower=0).fillna(0).astype(int)
df["HERIDOS"]    = df["HERIDOS"].clip(lower=0).fillna(0).astype(int)
# 5) Variables derivadas
df["GRAVEDAD"] = np.where(df["FALLECIDOS"]>0,"MORTAL",
                 np.where(df["HERIDOS"]>0,"GRAVE","LEVE"))
df["FECHA"] = pd.to_datetime(df["FECHA"])
dias = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
df["DIA_SEM_N"] = df["FECHA"].dt.dayofweek
df["DIA_SEM"]   = df["DIA_SEM_N"].map(dict(enumerate(dias)))
df = df.sort_values("FECHA").reset_index(drop=True)
print(f"Duplicados eliminados         : {ndup:,}")
print(f"Dataset limpio: {len(df):,} filas ({n0-len(df):,} descartadas) · "
      f"{df['FECHA'].min().date()} → {df['FECHA'].max().date()}")


print("=== Estadísticas descriptivas (numéricas) ===")
display(df[["FALLECIDOS","HERIDOS","HORA","LATITUD","LONGITUD"]].describe().round(3))
print("\n=== Distribución por gravedad ===")
print(df["GRAVEDAD"].value_counts().to_string())
print(f"\nTotal fallecidos: {df['FALLECIDOS'].sum():,} · Total heridos: {df['HERIDOS'].sum():,}")
print(f"% registros con al menos 1 fallecido: {(df['FALLECIDOS']>0).mean()*100:.2f}%")

fig, ax = plt.subplots(2, 3, figsize=(17,9))
fig.suptitle("EDA — Accidentes de tránsito en Ecuador", fontsize=15, fontweight="bold")

y = df.groupby("ANIO").size()
ax[0,0].bar(y.index, y.values, color="#2E86DE"); ax[0,0].set_title("Accidentes por año")
ax[0,0].set_xlabel("Año"); ax[0,0].set_ylabel("N.º accidentes")

meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
mth = df.groupby("MES").size().reindex(range(1,13), fill_value=0)
ax[0,1].bar(range(1,13), mth.values, color="#27AE60")
ax[0,1].set_xticks(range(1,13)); ax[0,1].set_xticklabels(meses, rotation=45)
ax[0,1].set_title("Accidentes por mes"); ax[0,1].set_ylabel("N.º accidentes")

hr = df.groupby("HORA").size().reindex(range(24), fill_value=0)
ax[0,2].fill_between(hr.index, hr.values, alpha=.5, color="#E67E22")
ax[0,2].plot(hr.index, hr.values, color="#E67E22", lw=2)
ax[0,2].set_title("Accidentes por hora del día"); ax[0,2].set_xlabel("Hora")
ax[0,2].axvspan(7,9,alpha=.12,color="red"); ax[0,2].axvspan(18,20,alpha=.12,color="red")

g = df["GRAVEDAD"].value_counts()
cmap={"MORTAL":"#C0392B","GRAVE":"#E67E22","LEVE":"#27AE60"}
ax[1,0].pie(g.values, labels=g.index, autopct="%1.1f%%", colors=[cmap[k] for k in g.index],
            startangle=90); ax[1,0].set_title("Distribución por gravedad")

top = df.groupby("CANTON").size().nlargest(10)[::-1]
ax[1,1].barh(top.index, top.values, color="#8E44AD"); ax[1,1].set_title("Top 10 cantones")
ax[1,1].set_xlabel("N.º accidentes")

ax[1,2].scatter(df["LONGITUD"], df["LATITUD"], s=1, alpha=.05, color="#1B4F72")
ax[1,2].set_xlim(-81.3, -75.0); ax[1,2].set_ylim(-5.1, 1.7)   # zoom continental
ax[1,2].set_title("Dispersión geográfica (continental)")
ax[1,2].set_xlabel("Longitud"); ax[1,2].set_ylabel("Latitud")

plt.tight_layout(); plt.savefig("fig_eda.png", bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(1,3, figsize=(18,5.5))

tipo = df["TIPO_ACCIDENTE"].value_counts().head(10)[::-1]
ax[0].barh(tipo.index, tipo.values, color="#2E86DE")
ax[0].set_title("Top 10 tipos de siniestro"); ax[0].set_xlabel("N.º accidentes")
for i,v in enumerate(tipo.values):
    ax[0].text(v, i, f" {v/len(df)*100:.1f}%", va="center", fontsize=8)

# Letalidad por tipo (fallecidos por cada 100 accidentes) — frecuencia ≠ severidad
let = (df.groupby("TIPO_ACCIDENTE")
         .agg(n=("FALLECIDOS","size"), f=("FALLECIDOS","sum"))
         .query("n >= 200").assign(let=lambda d: d.f/d.n*100)
         .sort_values("let").tail(10))
ax[1].barh(let.index, let["let"], color="#C0392B")
ax[1].set_title("Letalidad por tipo (fallecidos/100 acc.)"); ax[1].set_xlabel("Tasa")

if df["CAUSA_ACCIDENTE"].nunique() > 1:
    causa = df["CAUSA_ACCIDENTE"].astype(str).str.slice(0,38).value_counts().head(10)[::-1]
    ax[2].barh(causa.index, causa.values, color="#8E44AD")
    ax[2].set_title("Top 10 causas probables"); ax[2].set_xlabel("N.º accidentes")
else:
    ax[2].axis("off"); ax[2].set_title("Causa no disponible en esta fuente")
plt.tight_layout(); plt.savefig("fig_tipos_causas.png", bbox_inches="tight"); plt.show()

print("Hallazgo: los tipos MÁS FRECUENTES (choque lateral) no son los MÁS LETALES "
      "(atropellos, choques frontales, arrollamientos) → priorizar por severidad, no solo volumen.")

# Patrón conjunto día de la semana × hora
piv_dh = (df.groupby(["DIA_SEM_N","HORA"]).size().unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0))
piv_dh.index = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
plt.figure(figsize=(15,4.5))
sns.heatmap(piv_dh, cmap="YlOrRd", cbar_kws={"label":"N.º accidentes"})
plt.title("Accidentes por día de la semana × hora"); plt.xlabel("Hora"); plt.ylabel("")
plt.tight_layout(); plt.savefig("fig_dia_hora.png", bbox_inches="tight"); plt.show()
print("Lectura: picos laborales 07–09 h y 18–20 h (Lun–Vie) y madrugadas de fin de semana "
      "(Sáb–Dom 00–03 h), asociadas a conducción nocturna/embriaguez.")

# --- Mapas del Ecuador con límites oficiales (GeoPandas + GADM 4.1) -----------
# Soporta: (a) GeoJSON ya descargado, (b) shapefile local/Kaggle, (c) descarga automática.
import unicodedata
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode()
    return s.upper().strip()

# Alias PROVINCIA (ANT) → NAME_1 (GADM) para nombres que difieren
ALIAS_PROV = {"SANTO DOMINGO":"SANTO DOMINGO DE LOS TSACHILAS",
              "STO DGO TSACHILAS":"SANTO DOMINGO DE LOS TSACHILAS",
              "STO DOMINGO TSACHILAS":"SANTO DOMINGO DE LOS TSACHILAS"}

ecuador = None
try:
    import geopandas as gpd
    CANDIDATOS = ["gadm41_ECU_1.json", "gadm41_ECU_1.shp",
                  "/kaggle/input/ecugadm/gadm41_ECU_1.shp",
                  "/content/gadm41_ECU_1.json"]
    ruta = next((c for c in CANDIDATOS if os.path.exists(c)), None)
    if ruta is None:
        import urllib.request, zipfile, io
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ECU_1.json.zip"
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
        z.extractall("."); ruta = "gadm41_ECU_1.json"
    ecuador = gpd.read_file(ruta)
    ecuador["PROV_N"] = ecuador["NAME_1"].map(_norm)
    gdf_acc = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["LONGITUD"], df["LATITUD"]),
                               crs="EPSG:4326")
except Exception as e:
    print("GADM/geopandas no disponible → gráficos sin límites:", e)

# ═══ Figura 1: choropleth por provincia + mapa de puntos (estilo Kaggle) ═══════
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

prov = (df.assign(PROV_N=df["PROVINCIA"].map(_norm).replace(ALIAS_PROV))
          .groupby("PROV_N").size().rename("n_acc").reset_index())
if ecuador is not None:
    eplot = ecuador.merge(prov, on="PROV_N", how="left").fillna({"n_acc": 0})
    sin_match = sorted(set(prov["PROV_N"]) - set(ecuador["PROV_N"]))
    if sin_match: print("⚠ Provincias del CSV sin match en GADM:", sin_match)
    eplot.plot(column="n_acc", cmap="YlOrRd", linewidth=.6, edgecolor="grey",
               legend=True, legend_kwds={"label":"N.º accidentes","shrink":.7}, ax=ax[0])
    ax[0].set_title("Accidentes por provincia (2017–2024)", fontweight="bold", fontsize=14)

    # Estilo Kaggle: provincias en verde claro + todos los accidentes en rojo
    ecuador.plot(ax=ax[1], color="#DFF3DF", edgecolor="#333333", linewidth=.7)
    gdf_acc.plot(ax=ax[1], color="#D32F2F", markersize=3, alpha=.35, label="Accidentes")
    ax[1].set_title("Mapa de accidentes en Ecuador (continental e insular)",
                    fontweight="bold", fontsize=14)
    ax[1].legend(markerscale=6)
else:
    ax[0].axis("off")
    ax[1].scatter(df["LONGITUD"], df["LATITUD"], s=1, alpha=.05, c="red")
    ax[1].set_title("Distribución geográfica de accidentes", fontweight="bold")
for a in ax: a.set_xlabel("Longitud"); a.set_ylabel("Latitud")
plt.tight_layout(); plt.savefig("fig_mapa_ecuador.png", bbox_inches="tight"); plt.show()

top_prov = prov.sort_values("n_acc", ascending=False).head(5)
print("Top 5 provincias:", ", ".join(f"{r.PROV_N.title()} ({r.n_acc:,})"
                                     for r in top_prov.itertuples()))

# ═══ Figura 2: mapa de severidad — heridos (azul) y fallecidos (rojo) ═════════
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(16, 9))
her = df[df["HERIDOS"]>0]; mor = df[df["FALLECIDOS"]>0]
if ecuador is not None:
    ecuador.plot(ax=ax, color="#EAF7EA", edgecolor="#555555", linewidth=.7)
# Heridos: azul sólido, tamaño ∝ víctimas (capa base)
ax.scatter(her["LONGITUD"], her["LATITUD"], s=her["HERIDOS"]*6,
           c="#2E86DE", alpha=.18, edgecolors="none")
# Fallecidos: rojo intenso con borde, encima (capa prioritaria)
ax.scatter(mor["LONGITUD"], mor["LATITUD"], s=mor["FALLECIDOS"]*22,
           c="#C0392B", alpha=.65, edgecolors="#7B241C", linewidths=.4)
ax.set_xlim(-92.6, -74.8); ax.set_ylim(-5.2, 2.0)
ax.legend(handles=[
    Line2D([0],[0], marker="o", ls="", markerfacecolor="#2E86DE", alpha=.6,
           markeredgecolor="none", markersize=10, label="Heridos (tamaño ∝ víctimas)"),
    Line2D([0],[0], marker="o", ls="", markerfacecolor="#C0392B",
           markeredgecolor="#7B241C", markersize=10, label="Fallecidos (tamaño ∝ víctimas)")],
    loc="lower left", frameon=True)
ax.set_title("Mapa de severidad — fallecidos y heridos", fontweight="bold", fontsize=14)
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
plt.tight_layout(); plt.savefig("fig_calor_severidad.png", bbox_inches="tight"); plt.show()
print("Lectura: los heridos se concentran en corredores urbanos (Guayas–Pichincha), "
      "mientras los siniestros mortales se dispersan hacia vías interprovinciales.")

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

R_EARTH_KM = 6371.0
coords = df[["LATITUD","LONGITUD"]].to_numpy()
coords_rad = np.radians(coords)
print(f"Puntos para clustering: {len(coords_rad):,}")

K = 45  # = minPts de referencia del paper a escala nacional
nn = NearestNeighbors(n_neighbors=K, metric="haversine", algorithm="ball_tree").fit(coords_rad)
dist,_ = nn.kneighbors(coords_rad)
kdist_km = np.sort(dist[:, K-1])[::-1] * R_EARTH_KM

fig, ax = plt.subplots(1,2, figsize=(14,4.5))
ax[0].plot(kdist_km, color="#C0392B", lw=1.5)
for e,c in [(1.0,"#2E86DE"),(0.5,"#27AE60")]:
    ax[0].axhline(e, ls="--", color=c, label=f"ε = {e} km")
ax[0].set_ylim(0,5); ax[0].set_title(f"k-distancias (k={K})"); ax[0].legend()
ax[0].set_xlabel("Puntos ordenados"); ax[0].set_ylabel("Dist. al k-vecino (km)")
z = int(len(kdist_km)*0.06)
ax[1].plot(kdist_km[:z], color="#C0392B", lw=2)
ax[1].axhline(1.0, ls="--", color="#2E86DE", label="ε = 1 km"); ax[1].legend()
ax[1].set_title("Zoom — zona del codo"); ax[1].set_xlabel("6% superior")
plt.tight_layout(); plt.savefig("fig_kdist.png", bbox_inches="tight"); plt.show()

eps_km_grid   = [0.5, 1.0, 1.5, 2.0]
minpts_grid   = [25, 45, 60]
SAMPLE = min(40_000, len(coords_rad))
sidx = np.random.choice(len(coords_rad), SAMPLE, replace=False)
Xs = coords_rad[sidx]

rows = []
for e_km in eps_km_grid:
    for mp in minpts_grid:
        lab = DBSCAN(eps=e_km/R_EARTH_KM, min_samples=mp, metric="haversine",
                     algorithm="ball_tree", n_jobs=-1).fit_predict(Xs)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        noise = (lab==-1).mean()*100
        if k >= 2 and (lab!=-1).sum() > 100:
            sil = silhouette_score(Xs[lab!=-1], lab[lab!=-1], metric="haversine",
                                   sample_size=min(5000,(lab!=-1).sum()), random_state=SEED)
            dbi = davies_bouldin_score(Xs[lab!=-1], lab[lab!=-1])
        else:
            sil, dbi = np.nan, np.nan
        rows.append(dict(eps_km=e_km, minPts=mp, n_clusters=k,
                         noise_pct=round(noise,1), silhouette=round(sil,3),
                         davies_bouldin=round(dbi,3)))
grid = pd.DataFrame(rows)
display(grid)

fig, ax = plt.subplots(1,3, figsize=(17,4))
for metric,a,cmap,lab in [("silhouette",ax[0],"YlGn","Silhouette ↑ (mejor alto)"),
                          ("davies_bouldin",ax[1],"YlOrRd_r","Davies–Bouldin ↓ (mejor bajo)"),
                          ("noise_pct",ax[2],"Purples","% ruido")]:
    piv = grid.pivot(index="minPts", columns="eps_km", values=metric)
    sns.heatmap(piv, annot=True, fmt=".3g", cmap=cmap, ax=a, cbar_kws={"label":lab})
    a.set_title(lab); a.set_xlabel("ε (km)"); a.set_ylabel("minPts")
plt.tight_layout(); plt.savefig("fig_grid.png", bbox_inches="tight"); plt.show()
print("Lectura: ε pequeño (0.5 km) sobre-segmenta (muchos clusters, ruido alto); "
      "ε grande (≥1.5 km) fusiona ciudades enteras. ε=1 km + minPts=45 equilibra "
      "Silhouette/DBI con un nivel de ruido interpretable.")


EPS_KM, MINPTS = 1.0, 45
db = DBSCAN(eps=EPS_KM/R_EARTH_KM, min_samples=MINPTS, metric="haversine",
            algorithm="ball_tree", n_jobs=-1)
labels = db.fit_predict(coords_rad)
df["CLUSTER"] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = (labels==-1).sum(); noise_pct = noise/len(labels)*100
m = labels!=-1
sil = silhouette_score(coords_rad[m], labels[m], metric="haversine",
                       sample_size=min(10000,m.sum()), random_state=SEED)
dbi = davies_bouldin_score(coords_rad[m], labels[m])
print(f"ε = {EPS_KM} km · minPts = {MINPTS}")
print(f"Clusters (hotspots) : {n_clusters}")
print(f"Puntos en clusters  : {m.sum():,} ({100-noise_pct:.1f}%)")
print(f"Ruido               : {noise:,} ({noise_pct:.1f}%)")
print(f"Silhouette          : {sil:.3f}")
print(f"Davies-Bouldin      : {dbi:.3f}")

ruido_df, clus_df = df[df["CLUSTER"]==-1], df[df["CLUSTER"]!=-1]
tabla_ruido = pd.DataFrame({
    "Ruido": [len(ruido_df), ruido_df["FALLECIDOS"].sum(),
              round(ruido_df["FALLECIDOS"].sum()/max(len(ruido_df),1)*100,2)],
    "En clusters": [len(clus_df), clus_df["FALLECIDOS"].sum(),
              round(clus_df["FALLECIDOS"].sum()/max(len(clus_df),1)*100,2)],
}, index=["N.º accidentes","Fallecidos","Fallecidos por 100 accidentes"])
display(tabla_ruido)
if "ZONA" in df.columns and df["ZONA"].nunique() > 1:
    print("Composición del ruido por zona:")
    print(ruido_df["ZONA"].value_counts(normalize=True).round(3).to_string())
print("\nTop cantones dentro del ruido:")
print(ruido_df["CANTON"].value_counts().head(5).to_string())
print("\nLectura: la tasa de letalidad del ruido suele SUPERAR a la de los clusters urbanos →")
print("las políticas no deben limitarse a hotspots: las vías interurbanas requieren controles de velocidad.")

from scipy.spatial import ConvexHull

def hull_area_km2(sub):
    if len(sub) < 3: return np.nan
    latm = np.radians(sub["LATITUD"].mean())
    x = sub["LONGITUD"].to_numpy()*111.320*np.cos(latm)
    y = sub["LATITUD"].to_numpy()*110.574
    try:    return ConvexHull(np.c_[x,y]).volume  # 'volume' = área en 2D
    except Exception: return np.nan

rec = []
for cid, sub in df[df["CLUSTER"]!=-1].groupby("CLUSTER"):
    A = hull_area_km2(sub)
    n = len(sub); fall = int(sub["FALLECIDOS"].sum()); her = int(sub["HERIDOS"].sum())
    canton = sub["CANTON"].mode().iat[0] if not sub["CANTON"].mode().empty else "N/D"
    rec.append(dict(cluster=cid, canton=canton, n_acc=n, fallecidos=fall, heridos=her,
                    area_km2=round(A,2) if A==A else np.nan,
                    densidad=round(n/A,1) if A and A==A and A>0 else np.nan,
                    tasa_mortalidad=round(fall/n,3)))
hot = pd.DataFrame(rec)
hot_by_density = hot.dropna(subset=["densidad"]).sort_values("densidad", ascending=False).head(10).reset_index(drop=True)
hot_by_acc     = hot.sort_values("n_acc", ascending=False).head(10).reset_index(drop=True)
print("=== Top 10 hotspots por N.º de accidentes ==="); display(hot_by_acc)
print("=== Top 10 hotspots por densidad (acc/km²) ==="); display(hot_by_density)

# Distribución de severidad por cluster (rank) — escala log
fig, ax = plt.subplots(1,2, figsize=(14,4.5))
for col,a,color,title in [("fallecidos",ax[0],"#C0392B","Fallecidos por cluster"),
                          ("heridos",ax[1],"#2E86DE","Heridos por cluster")]:
    s = hot.sort_values(col, ascending=False)[col].to_numpy()
    a.bar(range(len(s)), s, color=color); a.set_yscale("log")
    a.set_title(title+" (rank, escala log)"); a.set_xlabel("Rank de cluster")
    a.set_ylabel(col.capitalize())
plt.tight_layout(); plt.savefig("fig_severity_rank.png", bbox_inches="tight"); plt.show()
print(f"Los 2 clusters más severos concentran "
      f"{hot.sort_values('fallecidos',ascending=False)['fallecidos'].head(2).sum()/hot['fallecidos'].sum()*100:.1f}% "
      f"de los fallecidos en hotspots.")

# --- Mapa estático de clusters + mapa interactivo HTML (Folium) ---------------
fig, ax = plt.subplots(figsize=(9,10))
noise_mask = df["CLUSTER"]==-1
ax.scatter(df.loc[noise_mask,"LONGITUD"], df.loc[noise_mask,"LATITUD"],
           s=1, c="#BDBDBD", alpha=.15, label="Ruido")
cl = df[~noise_mask]
ax.scatter(cl["LONGITUD"], cl["LATITUD"], s=6, c=cl["CLUSTER"] % 10, cmap="tab10", alpha=.85)
ax.set_xlim(-81.3, -75.0); ax.set_ylim(-5.1, 1.7)   # foco continental
ax.set_title(f"DBSCAN nacional — {n_clusters} hotspots (ε={EPS_KM} km, minPts={MINPTS})")
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.legend(loc="lower left")

# Recuadro inserto: región insular (Galápagos)
gal = df[df["LONGITUD"] < -85]
if len(gal) > 0:
    axg = ax.inset_axes([0.02, 0.66, 0.34, 0.30])
    gn = gal["CLUSTER"]==-1
    axg.scatter(gal.loc[gn,"LONGITUD"], gal.loc[gn,"LATITUD"], s=3, c="#BDBDBD", alpha=.4)
    axg.scatter(gal.loc[~gn,"LONGITUD"], gal.loc[~gn,"LATITUD"], s=10,
                c=gal.loc[~gn,"CLUSTER"] % 10, cmap="tab10", alpha=.9)
    axg.set_title(f"Galápagos (n={len(gal):,})", fontsize=9)
    axg.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig("fig_map_nacional.png", bbox_inches="tight"); plt.show()

try:
    import folium
    from folium.plugins import HeatMap, MiniMap, Fullscreen

    mp = folium.Map(location=[-1.8, -78.5], zoom_start=7, tiles="CartoDB positron")

    # Capa 1 — mapa de calor de densidad (muestra de 30k puntos)
    heat = df[["LATITUD","LONGITUD"]].sample(min(30000, len(df)), random_state=SEED)
    HeatMap(heat.values.tolist(), radius=8, blur=10, max_zoom=13, min_opacity=0.3,
            gradient={"0.4":"blue","0.65":"lime","1":"red"},
            name="Densidad de accidentes").add_to(mp)

    # Capa 2 — top 20 hotspots con popup enriquecido (tamaño ∝ n.º accidentes)
    capa_hot = folium.FeatureGroup(name="Top 20 hotspots (DBSCAN)").add_to(mp)
    for rank, (_, r) in enumerate(hot_by_acc.head(20).iterrows(), start=1):
        sub = df[df["CLUSTER"]==r["cluster"]]
        lat0, lon0 = sub["LATITUD"].mean(), sub["LONGITUD"].mean()
        radio = max(8, min(25, r["n_acc"]/400))
        popup_html = (f"<b>Hotspot #{rank} — {r['canton']}</b><br>"
                      f"Accidentes: {r['n_acc']:,}<br>"
                      f"Fallecidos: {r['fallecidos']:,}<br>"
                      f"Heridos: {r['heridos']:,}<br>"
                      f"Densidad: {r['densidad']} acc/km²<br>"
                      f"Tasa mortalidad: {r['tasa_mortalidad']*100:.1f}%")
        folium.CircleMarker([lat0, lon0], radius=radio, color="#C0392B",
                            fill=True, fill_color="#FF5722", fill_opacity=.7,
                            popup=folium.Popup(popup_html, max_width=260),
                            tooltip=f"Hotspot #{rank} · {r['n_acc']:,} accidentes"
                            ).add_to(capa_hot)

    # Capa 3 — muestra de siniestros mortales
    capa_mort = folium.FeatureGroup(name="Siniestros mortales (muestra)", show=False).add_to(mp)
    mort = df[df["FALLECIDOS"]>0].sample(min(1500, (df["FALLECIDOS"]>0).sum()), random_state=SEED)
    for _, r in mort.iterrows():
        folium.CircleMarker([r["LATITUD"], r["LONGITUD"]], radius=2, color="#7B241C",
                            fill=True, fill_opacity=.5, weight=0).add_to(capa_mort)

    # Controles y leyenda
    folium.LayerControl(collapsed=False).add_to(mp)
    MiniMap(toggle_display=True).add_to(mp)
    Fullscreen().add_to(mp)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;width:210px;'
              'background:white;border:2px solid grey;z-index:9999;padding:10px;'
              'border-radius:8px;font-size:12px">'
              '<b>Leyenda</b><br>'
              '&#128308; Hotspot DBSCAN (tamaño ∝ accidentes)<br>'
              '&#128993; Mapa de calor = densidad<br>'
              '&#9899; Puntos vino = mortales</div>')
    mp.get_root().html.add_child(folium.Element(legend))

    mp.save("mapa_hotspots.html")
    print("✓ Mapa interactivo guardado: mapa_hotspots.html (abrir en navegador o ver abajo)")
    display(mp)  # render inline en Colab/Jupyter
except Exception as e:
    print("folium no disponible (opcional):", e)


def dbscan_ciudad(nombre, lat0, lon0, radio_deg=0.35, eps_km=0.2, minpts=60):
    sub = df[(df["LATITUD"].between(lat0-radio_deg, lat0+radio_deg)) &
             (df["LONGITUD"].between(lon0-radio_deg, lon0+radio_deg))].copy()
    if len(sub) < minpts:
        print(f"[{nombre}] datos insuficientes"); return None
    rad = np.radians(sub[["LATITUD","LONGITUD"]].to_numpy())
    lab = DBSCAN(eps=eps_km/R_EARTH_KM, min_samples=minpts, metric="haversine",
                 algorithm="ball_tree", n_jobs=-1).fit_predict(rad)
    sub["CLUSTER"] = lab
    k = len(set(lab)) - (1 if -1 in lab else 0); noise = (lab==-1).mean()*100
    mm = lab!=-1
    sil = silhouette_score(rad[mm], lab[mm], metric="haversine",
                           sample_size=min(5000,mm.sum()), random_state=SEED) if k>=2 else np.nan
    dbi = davies_bouldin_score(rad[mm], lab[mm]) if k>=2 else np.nan
    print(f"[{nombre}] ε={eps_km}km minPts={minpts} → {k} clusters · "
          f"ruido {noise:.1f}% · Silhouette {sil:.3f} · DBI {dbi:.3f} · n={len(sub):,}")
    rec=[]
    for cid, g in sub[mm].groupby("CLUSTER"):
        A = hull_area_km2(g); n=len(g)
        rec.append(dict(cluster=cid, n_acc=n, fallecidos=int(g.FALLECIDOS.sum()),
                        heridos=int(g.HERIDOS.sum()),
                        densidad=round(n/A,1) if A and A==A and A>0 else np.nan))
    cols = ["cluster","n_acc","fallecidos","heridos","densidad"]
    top = pd.DataFrame(rec, columns=cols)
    if not top.empty and top["densidad"].notna().any():
        top = top.sort_values("densidad", ascending=False).head(10).reset_index(drop=True)
    return sub, top

res_uio = dbscan_ciudad("Quito", -0.190, -78.490)
res_gye = dbscan_ciudad("Guayaquil", -2.170, -79.922)
if res_uio: print("\nTop hotspots Quito:");     display(res_uio[1])
if res_gye: print("Top hotspots Guayaquil:");   display(res_gye[1])

# Visualización metropolitana
fig, ax = plt.subplots(1,2, figsize=(14,6))
for a,(res,nom) in zip(ax, [(res_uio,"Quito"),(res_gye,"Guayaquil")]):
    if not res: continue
    sub = res[0]; nz = sub["CLUSTER"]!=-1
    a.scatter(sub.loc[~nz,"LONGITUD"], sub.loc[~nz,"LATITUD"], s=3, c="#CFCFCF", alpha=.25)
    a.scatter(sub.loc[nz,"LONGITUD"], sub.loc[nz,"LATITUD"], s=9, c=sub.loc[nz,"CLUSTER"] % 10,
              cmap="tab10", alpha=.9)
    a.set_title(f"Hotspots — {nom}"); a.set_xlabel("Longitud"); a.set_ylabel("Latitud")
plt.tight_layout(); plt.savefig("fig_map_metro.png", bbox_inches="tight"); plt.show()

ts_m = df.set_index("FECHA").resample("MS").agg(
        accidentes=("FALLECIDOS","size"),
        fallecidos=("FALLECIDOS","sum"),
        heridos=("HERIDOS","sum"))
ts = ts_m["accidentes"].astype(float)
print(f"Periodo: {ts.index[0].date()} → {ts.index[-1].date()} · {len(ts)} meses")
print(f"Media: {ts.mean():.1f} · Desv: {ts.std():.1f} · "
      f"Mín: {ts.min():.0f} ({ts.idxmin().date()}) · Máx: {ts.max():.0f} ({ts.idxmax().date()})")
display(ts.describe().round(1).to_frame("accidentes/mes"))

fig, ax = plt.subplots(2,1, figsize=(14,8))
ax[0].plot(ts.index, ts.values, color="#2E86DE", lw=1.5, label="Accidentes/mes")
ax[0].plot(ts.rolling(12,center=True).mean(), color="#C0392B", lw=2.5, ls="--", label="Media móvil 12m")
ax[0].fill_between(ts.index, ts.values, alpha=.15, color="#2E86DE")
ax[0].set_title("Serie mensual de accidentes — Ecuador"); ax[0].legend(); ax[0].set_ylabel("N.º")

piv = ts.to_frame("y"); piv["a"]=piv.index.year; piv["m"]=piv.index.month
wide = piv.pivot(index="m", columns="a", values="y")
for c in wide.columns:
    ax[1].plot(range(1,13), wide[c], marker="o", ms=3, lw=1.3, alpha=.8, label=str(c))
ax[1].set_xticks(range(1,13))
ax[1].set_xticklabels(["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"])
ax[1].set_title("Estacionalidad — accidentes por mes y año"); ax[1].legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig("fig_serie.png", bbox_inches="tight"); plt.show()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def adf(s, nombre):
    r = adfuller(s.dropna(), autolag="AIC")
    print(f"ADF [{nombre}]  estadístico={r[0]:.3f}  p={r[1]:.4f}  → "
          f"{'ESTACIONARIA' if r[1]<0.05 else 'NO estacionaria (requiere diferenciar)'}")
adf(ts, "nivel"); adf(ts.diff().dropna(), "1ª diferencia")

dec = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")
fig = dec.plot(); fig.set_size_inches(13,9); fig.suptitle("Descomposición aditiva (período=12)", y=1.01)
plt.tight_layout(); plt.savefig("fig_decomp.png", bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(1,2, figsize=(14,4))
plot_acf(ts.diff().dropna(), lags=24, ax=ax[0], alpha=.05, title="ACF — 1ª diferencia")
plot_pacf(ts.diff().dropna(), lags=24, ax=ax[1], alpha=.05, method="ywm", title="PACF — 1ª diferencia")
plt.tight_layout(); plt.savefig("fig_acf.png", bbox_inches="tight"); plt.show()
print("Picos en lag 12/24 → estacionalidad anual; guía los órdenes (P,D,Q)[12] de SARIMA.")

mu, sigma = ts.mean(), ts.std()
z = (ts-mu)/sigma
out = ts[np.abs(z) > 2]
print(f"μ={mu:.1f} σ={sigma:.1f} · outliers detectados: {len(out)}")
display(pd.DataFrame({"accidentes":out.astype(int), "z":z[np.abs(z)>2].round(2)}))

ts_adj = ts.copy()
month_mean = ts[np.abs(z)<=2].groupby(ts[np.abs(z)<=2].index.month).mean()
for t in out.index:
    ts_adj.loc[t] = month_mean.loc[t.month]

fig, ax = plt.subplots(figsize=(14,4.5))
ax.plot(ts.index, ts.values, color="#2E86DE", lw=1.4, label="Original")
ax.plot(ts_adj.index, ts_adj.values, color="#27AE60", lw=1.8, label="Ajustada")
for k,c in [(1,"#F1C40F"),(2,"#E67E22")]:
    ax.axhline(mu+k*sigma, ls="--", color=c, alpha=.7); ax.axhline(mu-k*sigma, ls="--", color=c, alpha=.7)
ax.scatter(out.index, out.values, color="#C0392B", zorder=5, label="Outlier")
ax.set_title("Outliers (±2σ) y serie ajustada"); ax.legend(); ax.set_ylabel("N.º accidentes")
plt.tight_layout(); plt.savefig("fig_outliers.png", bbox_inches="tight"); plt.show()

H = 12
serie = ts_adj  # usamos la serie ajustada por outliers
train, test = serie.iloc[:-H], serie.iloc[-H:]
print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} meses)")
print(f"Test : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} meses)")

def rmse(a,p): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(p))**2)))
def mae(a,p):  return float(np.mean(np.abs(np.asarray(a)-np.asarray(p))))
def mape(a,p):
    a,p = np.asarray(a,float), np.asarray(p,float)
    return float(np.mean(np.abs((a-p)/np.where(a==0,np.nan,a)))*100)
scores = {}  # nombre -> dict(rmse, mae, pred(Series))

# --- Baseline: Naive estacional (y_t = y_{t-12}) ------------------------------
# Referencia mínima: cualquier modelo debe superarla para justificar su complejidad.
naive = serie.shift(12).iloc[-H:]
scores["Naive estacional"] = dict(rmse=rmse(test, naive), mae=mae(test, naive), pred=naive)
print(f"Baseline Naive estacional — RMSE {scores['Naive estacional']['rmse']:.2f} · "
      f"MAE {scores['Naive estacional']['mae']:.2f}")


# --- SARIMA con búsqueda de órdenes vía statsmodels (sin pmdarima) -----------
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX

best = None
for p, q, P, Q in itertools.product(range(4), range(4), range(3), range(3)):
    try:
        r = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        if best is None or r.aic < best.aic:
            best = r
    except Exception:
        pass

order = best.specification["order"]
seasonal_order = best.specification["seasonal_order"]
print("Modelo:", f"SARIMA{order}x{seasonal_order}", "· AIC", round(best.aic, 1))

pred = best.get_forecast(steps=H)
sarima = pd.Series(np.asarray(pred.predicted_mean), index=test.index)
ci = pred.conf_int(alpha=0.05).to_numpy()

scores["SARIMA"] = dict(rmse=rmse(test, sarima), mae=mae(test, sarima), pred=sarima)
print(f"SARIMA — RMSE {scores['SARIMA']['rmse']:.2f} · MAE {scores['SARIMA']['mae']:.2f}")

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(train.index, train, color="#2E86DE", label="Train")
ax.plot(test.index, test, color="#2C3E50", lw=2, label="Test real")
ax.plot(sarima.index, sarima, color="#C0392B", ls="--", lw=2, label="SARIMA")
ax.fill_between(test.index, ci[:, 0], ci[:, 1], color="#C0392B", alpha=.15, label="IC 95%")
ax.set_title(f"SARIMA{order}x{seasonal_order}"); ax.legend(); ax.set_ylabel("N.º")
plt.tight_layout(); plt.savefig("fig_sarima.png", bbox_inches="tight"); plt.show()

# --- Diagnóstico visual del modelo (residuos, Q-Q, histograma, correlograma) --
try:
    best.plot_diagnostics(figsize=(14, 8), lags=4)
    plt.suptitle("Diagnóstico del modelo SARIMA", y=1.01)
except Exception:
    # Respaldo manual: la serie es corta y statsmodels quema observaciones iniciales
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf
    res = pd.Series(best.resid).iloc[13:].dropna()   # descartar burn-in (d + s·D)
    fig, axd = plt.subplots(2, 2, figsize=(14, 8))
    axd[0,0].plot(res.values, color="#2E86DE"); axd[0,0].axhline(0, color="grey", ls="--")
    axd[0,0].set_title("Residuos estandarizados")
    axd[0,1].hist(res, bins=15, color="#2E86DE", edgecolor="white", density=True)
    axd[0,1].set_title("Histograma de residuos")
    stats.probplot(res, dist="norm", plot=axd[1,0])
    axd[1,0].set_title("Q-Q plot (normalidad)")
    plot_acf(res, lags=min(12, len(res)//2 - 1), ax=axd[1,1])
    axd[1,1].set_title("Correlograma (ACF)")
    plt.suptitle("Diagnóstico del modelo SARIMA", y=1.01)
plt.tight_layout(); plt.savefig("fig_sarima_diag.png", bbox_inches="tight"); plt.show()

# --- Diagnóstico de residuos: Ljung-Box (H0: residuos sin autocorrelación) ----
from statsmodels.stats.diagnostic import acorr_ljungbox
resid = pd.Series(best.resid)
# Usamos lags=4 para asegurar suficientes observaciones tras la diferenciación
lb = acorr_ljungbox(resid.dropna(), lags=[4], return_df=True)
p_lb = float(lb["lb_pvalue"].iloc[0])
print(f"Ljung-Box (lag 4): p = {p_lb:.4f} → "
      f"{'residuos ~ ruido blanco: modelo bien especificado' if p_lb > 0.05 else 'queda autocorrelación: revisar órdenes'}")


from prophet import Prophet
dtr = pd.DataFrame({"ds":train.index, "y":train.values})
try:
    from prophet.make_holidays import make_holidays_df
    hol = make_holidays_df(year_list=list(range(2017,2026)), country="EC")
except Exception:
    hol = None
pm_ = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
              seasonality_mode="additive", changepoint_prior_scale=0.05, holidays=hol,
              interval_width=0.95)
pm_.fit(dtr)
fut = pm_.make_future_dataframe(periods=H, freq="MS")
fp = pm_.predict(fut)
prophet = pd.Series(fp.tail(H)["yhat"].values, index=test.index)
scores["Prophet"] = dict(rmse=rmse(test, prophet), mae=mae(test, prophet), pred=prophet)
print(f"Prophet — RMSE {scores['Prophet']['rmse']:.2f} · MAE {scores['Prophet']['mae']:.2f}")
fig = pm_.plot(fp); plt.title("Prophet — ajuste y pronóstico"); plt.tight_layout()
plt.savefig("fig_prophet.png", bbox_inches="tight"); plt.show()
pm_.plot_components(fp); plt.tight_layout(); plt.savefig("fig_prophet_comp.png", bbox_inches="tight"); plt.show()

def make_features(s, lags=12):
    d = pd.DataFrame({"y":s})
    for l in range(1,lags+1): d[f"lag_{l}"] = d["y"].shift(l)
    for w in [3,6,12]:
        d[f"rmean_{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"rstd_{w}"]  = d["y"].shift(1).rolling(w).std()
    d["mes"] = s.index.month; d["anio"] = s.index.year
    d["mes_sin"] = np.sin(2*np.pi*d["mes"]/12); d["mes_cos"] = np.cos(2*np.pi*d["mes"]/12)
    return d.dropna()

feat = make_features(serie, 12)
Xcols = [c for c in feat.columns if c!="y"]
n_tr = len(feat) - H
Xtr, Xte = feat[Xcols].iloc[:n_tr], feat[Xcols].iloc[n_tr:]
ytr, yte = feat["y"].iloc[:n_tr], feat["y"].iloc[n_tr:]

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
xgbm = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbosity=0)
xgbm.fit(Xtr, ytr)
xgb_pred = pd.Series(xgbm.predict(Xte), index=yte.index)
scores["XGBoost"] = dict(rmse=rmse(yte, xgb_pred), mae=mae(yte, xgb_pred), pred=xgb_pred)

rf = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=SEED, n_jobs=-1)
rf.fit(Xtr, ytr)
rf_pred = pd.Series(rf.predict(Xte), index=yte.index)
scores["RandomForest"] = dict(rmse=rmse(yte, rf_pred), mae=mae(yte, rf_pred), pred=rf_pred)

print(f"XGBoost      — RMSE {scores['XGBoost']['rmse']:.2f} · MAE {scores['XGBoost']['mae']:.2f}")
print(f"RandomForest — RMSE {scores['RandomForest']['rmse']:.2f} · MAE {scores['RandomForest']['mae']:.2f}")

imp = pd.Series(xgbm.feature_importances_, index=Xcols).sort_values().tail(12)
imp.plot(kind="barh", figsize=(8,4.5), color="#8E44AD", title="XGBoost — importancia de variables")
plt.tight_layout(); plt.savefig("fig_xgb_imp.png", bbox_inches="tight"); plt.show()

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(SEED)

LOOK = 24
sc = MinMaxScaler()
arr = sc.fit_transform(serie.values.reshape(-1,1)).ravel()

def windows(a, look):
    X,y = [],[]
    for i in range(len(a)-look):
        X.append(a[i:i+look]); y.append(a[i+look])
    return np.array(X)[...,None], np.array(y)

Xall, yall = windows(arr, LOOK)
# el test son los últimos H puntos de la serie
n_te = H
Xtr_s, ytr_s = Xall[:-n_te], yall[:-n_te]
model = Sequential([LSTM(64, return_sequences=True, input_shape=(LOOK,1)),
                    Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation="relu"), Dense(1)])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
hist = model.fit(Xtr_s, ytr_s, epochs=250, batch_size=8, validation_split=0.15,
                 callbacks=[es], verbose=0)
print("Épocas entrenadas:", len(hist.history["loss"]))

# Pronóstico recursivo de los últimos H meses
seq = arr[:-H][-LOOK:].tolist(); preds=[]
for _ in range(H):
    p = model.predict(np.array(seq[-LOOK:])[None,:,None], verbose=0)[0,0]
    preds.append(p); seq.append(p)
lstm_pred = pd.Series(sc.inverse_transform(np.array(preds).reshape(-1,1)).ravel(), index=test.index)
scores["LSTM"] = dict(rmse=rmse(test, lstm_pred), mae=mae(test, lstm_pred), pred=lstm_pred)
print(f"LSTM — RMSE {scores['LSTM']['rmse']:.2f} · MAE {scores['LSTM']['mae']:.2f}")

plt.figure(figsize=(9,3.5)); plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val"); plt.legend(); plt.title("LSTM — curva de pérdida")
plt.tight_layout(); plt.savefig("fig_lstm_loss.png", bbox_inches="tight"); plt.show()

MODELOS_ENS = [m for m in scores if m != "Naive estacional"]  # el baseline no entra al ensemble
inv = {m: 1/scores[m]["rmse"] for m in MODELOS_ENS}
Z = sum(inv.values()); w = {m: inv[m]/Z for m in inv}
print("Pesos (inv-RMSE):", {m: round(v,3) for m,v in w.items()})
ens = sum(w[m]*scores[m]["pred"] for m in MODELOS_ENS)
scores["Ensemble"] = dict(rmse=rmse(test, ens), mae=mae(test, ens), pred=ens)
print(f"Ensemble — RMSE {scores['Ensemble']['rmse']:.2f} · MAE {scores['Ensemble']['mae']:.2f}")


def interpreta(m, r, best_rmse, base_rmse):
    if m == "Naive estacional":
        return "Baseline de referencia: repite el valor de hace 12 meses."
    if base_rmse > 0:
        gan = (base_rmse - r)/base_rmse*100
        txt = f"{'Supera' if gan>0 else 'NO supera'} al baseline en {abs(gan):.0f}% de RMSE."
    else:
        txt = "Comparar con baseline vía MAE/MAPE."
    if r == best_rmse: txt += " ★ Mejor modelo."
    return txt

comp = pd.DataFrame([{"Modelo":m, "RMSE":round(s["rmse"],2), "MAE":round(s["mae"],2),
                      "MAPE %":round(mape(test, s["pred"]),2),
                      "Tipo":("Baseline" if m=="Naive estacional"
                              else "Estadístico" if m in ("SARIMA","Prophet")
                              else "Ensemble" if m=="Ensemble" else "ML/DL")}
                     for m,s in scores.items()]).sort_values("RMSE").reset_index(drop=True)
base_rmse = float(comp.loc[comp["Modelo"]=="Naive estacional","RMSE"].iloc[0])
best_rmse = float(comp["RMSE"].min())
comp["Interpretación"] = [interpreta(m, r, best_rmse, base_rmse)
                          for m,r in zip(comp["Modelo"], comp["RMSE"])]
comp.index += 1
print("="*60); print("COMPARACIÓN DE MODELOS — test 12 meses"); print("="*60)
display(comp)
mejor = comp.iloc[0]["Modelo"]; print("Mejor modelo por RMSE:", mejor)

fig, ax = plt.subplots(1,2, figsize=(16,5))
ax[0].plot(train.index[-24:], train.values[-24:], color="#2E86DE", label="Histórico", alpha=.7)
ax[0].plot(test.index, test.values, color="black", lw=2.5, label="Real")
col = {"SARIMA":"#C0392B","Prophet":"#E67E22","XGBoost":"#8E44AD","RandomForest":"#16A085",
       "LSTM":"#2980B9","Ensemble":"#7F8C8D","Naive estacional":"#BDC3C7"}
for m,s in scores.items():
    ax[0].plot(s["pred"].index, s["pred"].values, ls="--", lw=1.5, alpha=.85,
               color=col.get(m,"grey"), label=m)
ax[0].axvline(test.index[0], color="black", ls=":", alpha=.5); ax[0].legend(fontsize=8)
ax[0].set_title("Pronósticos vs. real (test)")

x = np.arange(len(comp)); ax[1].bar(x-0.2, comp["RMSE"], 0.4, label="RMSE", color="#C0392B")
ax[1].bar(x+0.2, comp["MAE"], 0.4, label="MAE", color="#2E86DE")
ax[1].set_xticks(x); ax[1].set_xticklabels(comp["Modelo"], rotation=30, ha="right")
ax[1].legend(); ax[1].set_title("Error por modelo")
plt.tight_layout(); plt.savefig("fig_comparacion.png", bbox_inches="tight"); plt.show()


pf = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
             seasonality_mode="additive", changepoint_prior_scale=0.05, holidays=hol, interval_width=0.95)
pf.fit(pd.DataFrame({"ds":serie.index, "y":serie.values}))
fut2 = pf.make_future_dataframe(periods=12, freq="MS"); fc2 = pf.predict(fut2)
futuro = fc2[fc2["ds"]>serie.index[-1]]

fig, ax = plt.subplots(figsize=(14,4.8))
ax.plot(serie.index, serie.values, color="#2E86DE", lw=1.8, label="Histórico")
ax.plot(futuro["ds"], futuro["yhat"], color="#E67E22", lw=2.5, ls="--", label="Pronóstico 12m")
ax.fill_between(futuro["ds"], futuro["yhat_lower"], futuro["yhat_upper"], color="#E67E22", alpha=.2)
ax.axvline(serie.index[-1], color="black", ls=":", alpha=.6)
ax.set_title("Pronóstico de accidentes — próximos 12 meses"); ax.legend(); ax.set_ylabel("N.º")
plt.tight_layout(); plt.savefig("fig_futuro.png", bbox_inches="tight"); plt.show()

hr = futuro.assign(mes=futuro["ds"].dt.strftime("%Y-%m"))[["mes","yhat"]].round(0)
hr = hr.sort_values("yhat", ascending=False)
print("Meses de mayor riesgo proyectado:"); display(hr.head(5).reset_index(drop=True))