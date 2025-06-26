import pandas as pd
import numpy as np
import talib 
import ccxt
from datetime import datetime
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class EstrategiaCrucesEMAs:
    def __init__(self, symbol='WLD/USDT', timeframe='1h', ema_rapida=12, ema_lenta=26):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ema_rapida = ema_rapida
        self.ema_lenta = ema_lenta
        self.exchange = ccxt.binance()

    def obtener_datos(self, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error obteniendo datos: {e}")
            return None

    def calcular_vwap(self, df):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_values = []
        cumulative_pv = 0
        cumulative_volume = 0

        for i in range(len(df)):
            pv = typical_price.iloc[i] * df['volume'].iloc[i]
            cumulative_pv += pv
            cumulative_volume += df['volume'].iloc[i]
            vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else typical_price.iloc[i]
            vwap_values.append(vwap)

        return pd.Series(vwap_values, index=df.index)

    def calcular_fractales(self, df):
        df['fractal_alto'] = df['high'][(df['high'].shift(2) < df['high']) & 
                                        (df['high'].shift(1) < df['high']) & 
                                        (df['high'].shift(-1) < df['high']) & 
                                        (df['high'].shift(-2) < df['high'])].fillna(0)

        df['fractal_bajo'] = df['low'][(df['low'].shift(2) > df['low']) & 
                                       (df['low'].shift(1) > df['low']) & 
                                       (df['low'].shift(-1) > df['low']) & 
                                       (df['low'].shift(-2) > df['low'])].fillna(0)

        return df

    def calcular_indicadores(self, df):
        df['ema_rapida'] = talib.EMA(df['close'].values, timeperiod=self.ema_rapida)
        df['ema_lenta'] = talib.EMA(df['close'].values, timeperiod=self.ema_lenta)
        df['vwap'] = self.calcular_vwap(df)
        df['diferencia_emas'] = df['ema_rapida'] - df['ema_lenta']
        df['precio_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100

        df['cruce_alcista_emas'] = (df['ema_rapida'] > df['ema_lenta']) & (df['ema_rapida'].shift(1) <= df['ema_lenta'].shift(1))
        df['cruce_bajista_emas'] = (df['ema_rapida'] < df['ema_lenta']) & (df['ema_rapida'].shift(1) >= df['ema_lenta'].shift(1))

        df['precio_sobre_vwap'] = df['close'] > df['vwap']
        df['cruce_alcista_vwap'] = (df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))
        df['cruce_bajista_vwap'] = (df['close'] < df['vwap']) & (df['close'].shift(1) >= df['vwap'].shift(1))

        df['cruce_alcista'] = df['cruce_alcista_emas'] & df['precio_sobre_vwap']
        df['cruce_bajista'] = df['cruce_bajista_emas'] & ~df['precio_sobre_vwap']

        df['tendencia'] = np.where((df['ema_rapida'] > df['ema_lenta']) & (df['close'] > df['vwap']), 'ALCISTA',
                          np.where((df['ema_rapida'] < df['ema_lenta']) & (df['close'] < df['vwap']), 'BAJISTA', 'NEUTRAL'))

        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)

        return df

    def preparar_datos_ia(self, df, pasos_adelante=3):
        df = self.calcular_fractales(df)
        df = self.calcular_indicadores(df)
        df = df.dropna()

        df['target'] = df['close'].shift(-pasos_adelante)
        df = df.dropna()

        features = df[['close', 'ema_rapida', 'ema_lenta', 'vwap', 'rsi', 'macd', 'macd_signal', 'fractal_alto', 'fractal_bajo']]
        X = features.values
        y = df['target'].values

        return X, y, df

    def entrenar_modelo_ia(self, df):
        X, y, df_limpio = self.preparar_datos_ia(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"📊 Error Cuadrático Medio del modelo: {mse:.4f}")

        return modelo, df_limpio

    def predecir_precio_futuro(self, modelo, df):
        df = self.calcular_fractales(df)
        df = self.calcular_indicadores(df)
        df = df.dropna()

        features = df[['close', 'ema_rapida', 'ema_lenta', 'vwap', 'rsi', 'macd', 'macd_signal', 'fractal_alto', 'fractal_bajo']]
        ultima_fila = features.iloc[-1].values.reshape(1, -1)

        prediccion = modelo.predict(ultima_fila)[0]
        print(f"🔮 Predicción de precio a futuro: ${prediccion:.4f}")
        return prediccion

    def ejecutar_estrategia(self):
        print(f"🚀 Ejecutando estrategia para {self.symbol}")
        print(f"📊 Timeframe: {self.timeframe}")
        print(f"📈 EMA Rápida: {self.ema_rapida}, EMA Lenta: {self.ema_lenta}")
        print("-" * 50)

        df = self.obtener_datos(limit=200)
        if df is None:
            return

        df = self.calcular_indicadores(df)
        modelo, df_limpio = self.entrenar_modelo_ia(df)
        self.predecir_precio_futuro(modelo, df_limpio)

        return df

def main():
    estrategia = EstrategiaCrucesEMAs(
        symbol='WLD/USDT',
        timeframe='1h',
        ema_rapida=12,
        ema_lenta=26
    )

    df = estrategia.ejecutar_estrategia()

    if df is not None:
        df.to_csv('datos_wld_estrategia_ia.csv', index=False)
        print("\n💾 Datos guardados en 'datos_wld_estrategia_ia.csv'")

if __name__ == "__main__":
    main()
