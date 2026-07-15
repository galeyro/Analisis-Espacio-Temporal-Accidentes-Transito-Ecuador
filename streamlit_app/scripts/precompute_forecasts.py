import os
import sys
import pandas as pd
import numpy as np
import itertools
import warnings

# Configuración de Warnings y Semillas
warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# Añadir el directorio padre al sys.path para poder importar config y data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.data_loader import load_and_clean_data

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

def rmse(a,p): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(p))**2)))
def mae(a,p):  return float(np.mean(np.abs(np.asarray(a)-np.asarray(p))))
def mape(a,p):
    a,p = np.asarray(a,float), np.asarray(p,float)
    return float(np.mean(np.abs((a-p)/np.where(a==0,np.nan,a)))*100)

def make_features(s, lags=12):
    d = pd.DataFrame({"y":s})
    for l in range(1,lags+1): d[f"lag_{l}"] = d["y"].shift(l)
    for w in [3,6,12]:
        d[f"rmean_{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"rstd_{w}"]  = d["y"].shift(1).rolling(w).std()
    d["mes"] = s.index.month; d["anio"] = s.index.year
    d["mes_sin"] = np.sin(2*np.pi*d["mes"]/12); d["mes_cos"] = np.cos(2*np.pi*d["mes"]/12)
    return d.dropna()

def windows(a, look):
    X,y = [],[]
    for i in range(len(a)-look):
        X.append(a[i:i+look]); y.append(a[i+look])
    return np.array(X)[...,None], np.array(y)

def main():
    print("Cargando datos reales...")
    df = load_and_clean_data(config.DATA_FILE)
    if df.empty:
        print("Error: No se pudo cargar el dataset.")
        return
        
    print("Preparando serie temporal...")
    # Crear serie temporal mensual continua
    ts_m = df.set_index("FECHA").resample("MS").agg(accidentes=("AÑO", "size"))
    ts = ts_m["accidentes"].astype(float)
    
    # -----------------------------------------------------
    # Ajuste de Outliers (Para evitar distorsiones por COVID-19 u otros)
    # -----------------------------------------------------
    mu, sigma = ts.mean(), ts.std()
    z = (ts-mu)/sigma
    out = ts[np.abs(z) > 2]
    ts_adj = ts.copy()
    
    # Suavizado por la media histórica de ese mes
    meses_validos = ts[np.abs(z) <= 2]
    month_mean = meses_validos.groupby(meses_validos.index.month).mean()
    
    for t in out.index:
        if t.month in month_mean:
            ts_adj.loc[t] = month_mean.loc[t.month]
        else:
            ts_adj.loc[t] = mu
            
    serie = ts_adj
    H = 12
    train, test = serie.iloc[:-H], serie.iloc[-H:]
    
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} meses)")
    print(f"Test : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} meses)")
    
    results = pd.DataFrame(index=test.index)
    results['Real'] = test.values
    scores = {}

    # ==========================
    # 0. Baseline (Naive Estacional)
    # ==========================
    print("Calculando Baseline Naive...")
    naive = serie.shift(12).iloc[-H:]
    results['Naive estacional'] = naive.values
    scores["Naive estacional"] = dict(rmse=rmse(test, naive), mae=mae(test, naive), pred=naive)

    # ==========================
    # 1. SARIMA (Grid Search)
    # ==========================
    print("Entrenando SARIMA (Búsqueda de hiperparámetros - Esto puede tardar unos minutos)...")
    try:
        best_sarima = None
        # Exploración del espacio paramétrico (p,d,q) y (P,D,Q)
        for p, q, P, Q in itertools.product(range(4), range(4), range(3), range(3)):
            try:
                r = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if best_sarima is None or r.aic < best_sarima.aic:
                    best_sarima = r
            except:
                continue
                
        print(f"Mejor SARIMA encontrado: {best_sarima.specification['order']}x{best_sarima.specification['seasonal_order']}")
        pred_sarima = best_sarima.get_forecast(steps=H).predicted_mean
        sarima_s = pd.Series(np.asarray(pred_sarima), index=test.index)
        results['SARIMA'] = sarima_s.values
        scores["SARIMA"] = dict(rmse=rmse(test, sarima_s), mae=mae(test, sarima_s), pred=sarima_s)
    except Exception as e:
        print(f"Error SARIMA: {e}")
        
    # ==========================
    # 2. Prophet (con Feriados EC)
    # ==========================
    print("Entrenando Prophet...")
    try:
        dtr = pd.DataFrame({"ds":train.index, "y":train.values})
        # Tratar de cargar feriados de Ecuador
        try:
            from prophet.make_holidays import make_holidays_df
            hol = make_holidays_df(year_list=list(range(2017,2026)), country="EC")
        except Exception:
            hol = None
            
        pm_ = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                      seasonality_mode="additive", changepoint_prior_scale=0.05, holidays=hol, interval_width=0.95)
        pm_.fit(dtr)
        fut = pm_.make_future_dataframe(periods=H, freq="MS")
        fp = pm_.predict(fut)
        prophet_s = pd.Series(fp.tail(H)["yhat"].values, index=test.index)
        results['Prophet'] = prophet_s.values
        scores["Prophet"] = dict(rmse=rmse(test, prophet_s), mae=mae(test, prophet_s), pred=prophet_s)
    except Exception as e:
        print(f"Error Prophet: {e}")

    # ==========================
    # 3. XGBoost & Random Forest
    # ==========================
    print("Entrenando XGBoost y Random Forest...")
    try:
        feat = make_features(serie, 12)
        Xcols = [c for c in feat.columns if c!="y"]
        n_tr = len(feat) - H
        Xtr, Xte = feat[Xcols].iloc[:n_tr], feat[Xcols].iloc[n_tr:]
        ytr, yte = feat["y"].iloc[:n_tr], feat["y"].iloc[n_tr:]
        
        # XGBoost
        xgbm = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbosity=0)
        xgbm.fit(Xtr, ytr)
        xgb_pred = pd.Series(xgbm.predict(Xte), index=yte.index)
        results['XGBoost'] = xgb_pred.values
        scores["XGBoost"] = dict(rmse=rmse(yte, xgb_pred), mae=mae(yte, xgb_pred), pred=xgb_pred)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=400, max_depth=8, random_state=SEED, n_jobs=-1)
        rf.fit(Xtr, ytr)
        rf_pred = pd.Series(rf.predict(Xte), index=yte.index)
        results['Random Forest'] = rf_pred.values
        scores["Random Forest"] = dict(rmse=rmse(yte, rf_pred), mae=mae(yte, rf_pred), pred=rf_pred)
    except Exception as e:
        print(f"Error XGBoost/RF: {e}")

    # ==========================
    # 4. LSTM (Deep Learning)
    # ==========================
    if HAS_TF:
        print("Entrenando LSTM (250 Épocas, EarlyStopping)...")
        try:
            tf.random.set_seed(SEED)
            LOOK = 24
            sc = MinMaxScaler()
            arr = sc.fit_transform(serie.values.reshape(-1,1)).ravel()
            
            Xall, yall = windows(arr, LOOK)
            n_te = H
            Xtr_s, ytr_s = Xall[:-n_te], yall[:-n_te]
            
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(LOOK,1)),
                Dropout(0.2), 
                LSTM(32), 
                Dropout(0.2), 
                Dense(16, activation="relu"), 
                Dense(1)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
            
            model.fit(Xtr_s, ytr_s, epochs=250, batch_size=8, validation_split=0.15, callbacks=[es], verbose=0)
            
            # Pronóstico recursivo (predicting steps forward 1 by 1)
            seq = arr[:-H][-LOOK:].tolist()
            preds = []
            for _ in range(H):
                p = model.predict(np.array(seq[-LOOK:])[None,:,None], verbose=0)[0,0]
                preds.append(p)
                seq.append(p)
                
            lstm_pred = pd.Series(sc.inverse_transform(np.array(preds).reshape(-1,1)).ravel(), index=test.index)
            results['LSTM'] = lstm_pred.values
            scores["LSTM"] = dict(rmse=rmse(test, lstm_pred), mae=mae(test, lstm_pred), pred=lstm_pred)
        except Exception as e:
            print(f"Error LSTM: {e}")
    else:
        print("Saltando LSTM por no tener TensorFlow instalado. Continúa con los demás modelos.")

    # ==========================
    # 5. Ensemble (Weighted por Inversa del RMSE)
    # ==========================
    print("Generando Ensemble...")
    # Solo modelos predictivos, sin el Baseline
    modelos_ens = [m for m in scores if m != "Naive estacional"]
    if len(modelos_ens) > 0:
        inv = {m: 1/scores[m]["rmse"] for m in modelos_ens}
        Z = sum(inv.values())
        w = {m: inv[m]/Z for m in inv}
        ens = sum(w[m]*scores[m]["pred"] for m in modelos_ens)
        results['Ensemble'] = ens.values
        scores["Ensemble"] = dict(rmse=rmse(test, ens), mae=mae(test, ens), pred=ens)
        
    # Compilar métricas finales
    metrics = []
    for m, s in scores.items():
        metrics.append({
            'Model': m,
            'RMSE': s['rmse'],
            'MAE': s['mae'],
            'MAPE': mape(test, s['pred'])
        })
    
    # Guardar resultados
    print(f"Guardando predicciones en {config.FORECAST_RESULTS_FILE}...")
    results.reset_index().rename(columns={'FECHA_MES':'FECHA_MES'}).to_csv(config.FORECAST_RESULTS_FILE, index=False)
    
    print(f"Guardando métricas en {config.FORECAST_METRICS_FILE}...")
    df_metrics = pd.DataFrame(metrics)
    # Ordenar por RMSE para que el mejor quede arriba
    df_metrics = df_metrics.sort_values('RMSE').reset_index(drop=True)
    df_metrics.to_csv(config.FORECAST_METRICS_FILE, index=False)
    
    print("¡Precomputación finalizada con éxito! Resultados:")
    print(df_metrics)

if __name__ == "__main__":
    main()
