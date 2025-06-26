import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime, timedelta
import time
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

class EstrategiaPredictivaAvanzada:
    def __init__(self, symbol='WLD/USDT', timeframe='1h', ema_rapida=12, ema_lenta=26, 
                 take_profit_target=20.0, min_confidence=0.75):
        """
        Estrategia avanzada con predicción de precios y take profit optimizado
        
        Args:
            symbol (str): Par de trading
            timeframe (str): Marco temporal
            ema_rapida (int): Período de EMA rápida
            ema_lenta (int): Período de EMA lenta
            take_profit_target (float): Objetivo de take profit en %
            min_confidence (float): Confianza mínima para ejecutar señal (0-1)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.ema_rapida = ema_rapida
        self.ema_lenta = ema_lenta
        self.take_profit_target = take_profit_target
        self.min_confidence = min_confidence
        self.exchange = ccxt.binance()
        
        # Modelos de predicción
        self.modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.modelo_lr = LinearRegression()
        self.scaler = StandardScaler()
        self.modelos_entrenados = False
        
    def obtener_datos_historicos(self, limit=500):
        """
        Obtiene datos históricos más extensos para entrenamiento
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error obteniendo datos: {e}")
            return None
    
    def crear_features_avanzadas(self, df):
        """
        Crea características avanzadas para el modelo predictivo
        """
        # Indicadores técnicos básicos
        df['ema_rapida'] = talib.EMA(df['close'].values, timeperiod=self.ema_rapida)
        df['ema_lenta'] = talib.EMA(df['close'].values, timeperiod=self.ema_lenta)
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
        
        # Bandas de Bollinger
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Indicadores de volumen
        df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP
        df['vwap'] = self.calcular_vwap(df)
        df['precio_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Indicadores de momentum
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)
        
        # Patrones de precios
        df['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Características de precio
        df['precio_change'] = df['close'].pct_change()
        df['precio_change_5'] = df['close'].pct_change(5)
        df['volatilidad'] = df['precio_change'].rolling(20).std()
        
        # Features de tiempo
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Niveles de soporte y resistencia
        df['soporte'] = df['low'].rolling(20).min()
        df['resistencia'] = df['high'].rolling(20).max()
        df['distancia_soporte'] = (df['close'] - df['soporte']) / df['close']
        df['distancia_resistencia'] = (df['resistencia'] - df['close']) / df['close']
        
        return df
    
    def calcular_vwap(self, df):
        """Calcula VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_values = []
        cumulative_pv = 0
        cumulative_volume = 0
        
        for i in range(len(df)):
            pv = typical_price.iloc[i] * df['volume'].iloc[i]
            cumulative_pv += pv
            cumulative_volume += df['volume'].iloc[i]
            
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
            else:
                vwap = typical_price.iloc[i]
            
            vwap_values.append(vwap)
        
        return pd.Series(vwap_values, index=df.index)
    
    def preparar_datos_entrenamiento(self, df, ventana_prediccion=10):
        """
        Prepara datos para entrenamiento del modelo predictivo
        """
        # Seleccionar features para el modelo
        feature_columns = [
            'ema_rapida', 'ema_lenta', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position', 'volume_ratio', 'precio_vs_vwap',
            'stoch_k', 'stoch_d', 'williams_r', 'cci', 'precio_change',
            'precio_change_5', 'volatilidad', 'hour', 'day_of_week',
            'distancia_soporte', 'distancia_resistencia'
        ]
        
        # Eliminar NaN y preparar features
        df_clean = df[feature_columns + ['close']].dropna()
        
        X = []
        y = []
        
        # Crear ventanas de datos para predicción
        for i in range(len(df_clean) - ventana_prediccion):
            X.append(df_clean[feature_columns].iloc[i].values)
            # Precio máximo en las próximas ventana_prediccion velas
            precio_actual = df_clean['close'].iloc[i]
            precio_max_futuro = df_clean['close'].iloc[i+1:i+ventana_prediccion+1].max()
            ganancia_maxima = (precio_max_futuro - precio_actual) / precio_actual * 100
            y.append(ganancia_maxima)
        
        return np.array(X), np.array(y), feature_columns
    
    def entrenar_modelos(self, df):
        """
        Entrena los modelos de predicción
        """
        print("🤖 Entrenando modelos de predicción...")
        
        X, y, feature_names = self.preparar_datos_entrenamiento(df)
        
        if len(X) < 50:
            print("❌ No hay suficientes datos para entrenamiento")
            return False
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar Random Forest
        self.modelo_rf.fit(X_train, y_train)
        y_pred_rf = self.modelo_rf.predict(X_test)
        r2_rf = r2_score(y_test, y_pred_rf)
        
        # Entrenar Regresión Lineal
        self.modelo_lr.fit(X_train_scaled, y_train)
        y_pred_lr = self.modelo_lr.predict(X_test_scaled)
        r2_lr = r2_score(y_test, y_pred_lr)
        
        print(f"📊 R² Random Forest: {r2_rf:.3f}")
        print(f"📊 R² Regresión Lineal: {r2_lr:.3f}")
        
        self.feature_names = feature_names
        self.modelos_entrenados = True
        return True
    
    def predecir_ganancia_maxima(self, features):
        """
        Predice la ganancia máxima usando ensemble de modelos
        """
        if not self.modelos_entrenados:
            return 0, 0
        
        # Predicción con Random Forest
        pred_rf = self.modelo_rf.predict([features])[0]
        
        # Predicción con Regresión Lineal
        features_scaled = self.scaler.transform([features])
        pred_lr = self.modelo_lr.predict(features_scaled)[0]
        
        # Promedio ponderado (RF tiene más peso)
        prediccion_ensemble = (pred_rf * 0.7) + (pred_lr * 0.3)
        
        # Calcular confianza basada en importancia de features (RF)
        importancias = self.modelo_rf.feature_importances_
        confianza = np.sum(importancias * np.abs(features)) / np.sum(np.abs(features))
        
        return prediccion_ensemble, min(confianza, 1.0)
    
    def calcular_probabilidad_take_profit(self, df, precio_actual):
        """
        Calcula la probabilidad histórica de alcanzar el take profit
        """
        # Analizar patrones históricos similares
        contexto_actual = self.obtener_contexto_mercado(df.iloc[-1])
        
        # Buscar situaciones similares en el historial
        situaciones_similares = []
        for i in range(50, len(df) - 10):  # Dejar margen para análisis futuro
            contexto_historico = self.obtener_contexto_mercado(df.iloc[i])
            
            # Calcular similitud
            similitud = self.calcular_similitud_contexto(contexto_actual, contexto_historico)
            
            if similitud > 0.7:  # Si es suficientemente similar
                # Verificar si alcanzó take profit en las siguientes 10 velas
                precio_entrada = df.iloc[i]['close']
                precios_futuros = df.iloc[i+1:i+11]['high']
                ganancia_maxima = (precios_futuros.max() - precio_entrada) / precio_entrada * 100
                
                situaciones_similares.append({
                    'similitud': similitud,
                    'ganancia_maxima': ganancia_maxima,
                    'alcanzo_target': ganancia_maxima >= self.take_profit_target
                })
        
        if len(situaciones_similares) > 5:
            alcanzaron_target = sum([s['alcanzo_target'] for s in situaciones_similares])
            probabilidad = alcanzaron_target / len(situaciones_similares)
            ganancia_promedio = np.mean([s['ganancia_maxima'] for s in situaciones_similares])
            return probabilidad, ganancia_promedio
        
        return 0.5, 0  # Valor neutro si no hay suficientes datos
    
    def obtener_contexto_mercado(self, fila):
        """
        Obtiene el contexto actual del mercado
        """
        return {
            'rsi': fila.get('rsi', 50),
            'macd_hist': fila.get('macd_hist', 0),
            'bb_position': fila.get('bb_position', 0.5),
            'volume_ratio': fila.get('volume_ratio', 1),
            'precio_vs_vwap': fila.get('precio_vs_vwap', 0),
            'volatilidad': fila.get('volatilidad', 0),
            'tendencia_emas': 1 if fila.get('ema_rapida', 0) > fila.get('ema_lenta', 0) else -1
        }
    
    def calcular_similitud_contexto(self, contexto1, contexto2):
        """
        Calcula similitud entre dos contextos de mercado
        """
        diferencias = []
        for key in contexto1.keys():
            if key in contexto2:
                if key == 'tendencia_emas':
                    diferencias.append(0 if contexto1[key] == contexto2[key] else 1)
                else:
                    val1 = contexto1[key] if not pd.isna(contexto1[key]) else 0
                    val2 = contexto2[key] if not pd.isna(contexto2[key]) else 0
                    diferencias.append(abs(val1 - val2) / (abs(val1) + abs(val2) + 1))
        
        similitud = 1 - (np.mean(diferencias) if diferencias else 1)
        return max(0, similitud)
    
    def generar_señales_avanzadas(self, df):
        """
        Genera señales con análisis predictivo avanzado
        """
        if not self.modelos_entrenados:
            print("❌ Modelos no entrenados")
            return []
        
        señales = []
        
        for i in range(len(df) - 10):  # Dejar margen para análisis
            fila = df.iloc[i]
            
            # Condiciones básicas para señal alcista
            if (fila['ema_rapida'] > fila['ema_lenta'] and 
                fila['close'] > fila['vwap'] and
                fila['rsi'] < 70 and fila['rsi'] > 30):
                
                # Preparar features para predicción
                features = []
                for col in self.feature_names:
                    val = fila.get(col, 0)
                    features.append(val if not pd.isna(val) else 0)
                
                # Predicción de ganancia máxima
                ganancia_pred, confianza_modelo = self.predecir_ganancia_maxima(features)
                
                # Probabilidad histórica
                prob_tp, ganancia_historica = self.calcular_probabilidad_take_profit(df.iloc[:i+1], fila['close'])
                
                # Calcular score de confianza total
                score_confianza = (confianza_modelo * 0.4) + (prob_tp * 0.6)
                
                # Filtros adicionales para alta confianza
                filtros_ok = 0
                total_filtros = 5
                
                # Filtro 1: Predicción optimista
                if ganancia_pred >= self.take_profit_target * 0.8:
                    filtros_ok += 1
                
                # Filtro 2: Probabilidad histórica alta
                if prob_tp >= 0.6:
                    filtros_ok += 1
                
                # Filtro 3: Momentum positivo
                if fila['macd'] > fila['macd_signal'] and fila['macd_hist'] > 0:
                    filtros_ok += 1
                
                # Filtro 4: Posición en Bollinger Bands
                if 0.2 <= fila.get('bb_position', 0.5) <= 0.8:
                    filtros_ok += 1
                
                # Filtro 5: Volumen elevado
                if fila.get('volume_ratio', 1) > 1.2:
                    filtros_ok += 1
                
                score_filtros = filtros_ok / total_filtros
                score_final = (score_confianza * 0.7) + (score_filtros * 0.3)
                
                # Solo generar señal si cumple criterios estrictos
                if (score_final >= self.min_confidence and 
                    ganancia_pred >= self.take_profit_target * 0.7 and
                    prob_tp >= 0.5):
                    
                    señal = {
                        'tipo': 'COMPRA',
                        'datetime': fila['datetime'],
                        'precio': fila['close'],
                        'prediccion_ganancia': ganancia_pred,
                        'probabilidad_tp': prob_tp,
                        'score_confianza': score_final,
                        'ganancia_historica': ganancia_historica,
                        'ema_rapida': fila['ema_rapida'],
                        'ema_lenta': fila['ema_lenta'],
                        'rsi': fila['rsi'],
                        'vwap': fila['vwap'],
                        'recomendacion': 'ALTA_CONFIANZA' if score_final > 0.8 else 'CONFIANZA_MEDIA'
                    }
                    señales.append(señal)
        
        return señales
    
    def ejecutar_estrategia_completa(self):
        """
        Ejecuta la estrategia completa con predicción
        """
        print(f"🚀 ESTRATEGIA PREDICTIVA AVANZADA - {self.symbol}")
        print(f"🎯 Objetivo Take Profit: {self.take_profit_target}%")
        print(f"🔍 Confianza Mínima: {self.min_confidence*100}%")
        print("-" * 60)
        
        # Obtener datos históricos
        df = self.obtener_datos_historicos(limit=500)
        if df is None:
            return None, [], None
        
        # Crear features avanzadas
        df = self.crear_features_avanzadas(df)
        
        # Entrenar modelos
        if not self.entrenar_modelos(df):
            return df, [], None
        
        # Generar señales avanzadas
        señales = self.generar_señales_avanzadas(df)
        
        # Mostrar señales de alta confianza
        print("🎯 SEÑALES DE ALTA CONFIANZA:")
        señales_alta = [s for s in señales if s['score_confianza'] > 0.8]
        
        for señal in señales_alta[-3:]:  # Últimas 3 señales de alta confianza
            print(f"  📈 {señal['datetime']} - Precio: ${señal['precio']:.4f}")
            print(f"     Predicción: +{señal['prediccion_ganancia']:.2f}%")
            print(f"     Probabilidad TP: {señal['probabilidad_tp']*100:.1f}%")
            print(f"     Score: {señal['score_confianza']:.3f}")
            print(f"     RSI: {señal['rsi']:.1f}")
            print()
        
        # Análisis actual
        print("📊 ANÁLISIS ACTUAL:")
        ultimo = df.iloc[-1]
        features_actuales = []
        for col in self.feature_names:
            val = ultimo.get(col, 0)
            features_actuales.append(val if not pd.isna(val) else 0)
        
        pred_actual, conf_actual = self.predecir_ganancia_maxima(features_actuales)
        prob_actual, _ = self.calcular_probabilidad_take_profit(df, ultimo['close'])
        
        print(f"  Precio Actual: ${ultimo['close']:.4f}")
        print(f"  Predicción Ganancia: +{pred_actual:.2f}%")
        print(f"  Probabilidad TP 20%: {prob_actual*100:.1f}%")
        print(f"  Confianza Modelo: {conf_actual:.3f}")
        print(f"  RSI: {ultimo['rsi']:.1f}")
        print(f"  Tendencia: {'ALCISTA' if ultimo['ema_rapida'] > ultimo['ema_lenta'] else 'BAJISTA'}")
        
        return df, señales, {
            'prediccion_actual': pred_actual,
            'probabilidad_actual': prob_actual,
            'confianza_actual': conf_actual,
            'señales_alta_confianza': len(señales_alta)
        }

def main():
    """
    Función principal mejorada
    """
    estrategia = EstrategiaPredictivaAvanzada(
        symbol='WLD/USDT',
        timeframe='1h',
        ema_rapida=12,
        ema_lenta=26,
        take_profit_target=20.0,  # 20% objetivo
        min_confidence=0.75       # 75% confianza mínima
    )
    
    df, señales, analisis = estrategia.ejecutar_estrategia_completa()
    
    if df is not None:
        # Guardar resultados
        df.to_csv('datos_estrategia_predictiva.csv', index=False)
        
        # Guardar señales
        if señales:
            señales_df = pd.DataFrame(señales)
            señales_df.to_csv('señales_predictivas.csv', index=False)
            print(f"\n💾 {len(señales)} señales guardadas en 'señales_predictivas.csv'")
        
        print("\n💾 Datos guardados en 'datos_estrategia_predictiva.csv'")
        
        if analisis and analisis['prediccion_actual'] >= 15:
            print(f"\n🔥 ¡OPORTUNIDAD DETECTADA!")
            print(f"   Predicción: +{analisis['prediccion_actual']:.2f}%")
            print(f"   Probabilidad: {analisis['probabilidad_actual']*100:.1f}%")

if __name__ == "__main__":
    main()