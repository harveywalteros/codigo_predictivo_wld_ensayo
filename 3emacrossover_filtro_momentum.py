import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
import ccxt # Importar la librería ccxt
warnings.filterwarnings('ignore')

# --- 1. Configuración del Exchange y Credenciales (Bitget) ---
# ¡ADVERTENCIA! NUNCA PONGAS TUS CREDENCIALES DIRECTAMENTE EN EL CÓDIGO EN PRODUCCIÓN.
# Usa variables de entorno o un archivo de configuración seguro.
# Para pruebas, puedes descomentar y rellenar.
BITGET_API_KEY = 'TU_API_KEY_BITGET'
BITGET_SECRET = 'TU_SECRET_BITGET'

exchange = ccxt.bitget({
    'apiKey': BITGET_API_KEY,
    'secret': BITGET_SECRET,
    'options': {
        'defaultType': 'future', # O 'spot' si operas en el mercado al contado
    },
    'enableRateLimit': True, # Para evitar exceder los límites de la API
})

# Configurar timezone de Bogotá
timezone = pytz.timezone('America/Bogota')

# --- 2. Funciones de Cálculo de Indicadores Técnicos ---

def calcular_ema(precios, periodo):
    """Calcula la Media Móvil Exponencial (EMA)."""
    return precios.ewm(span=periodo, adjust=False).mean()

def calcular_macd(precios, fast_period=12, slow_period=26, signal_period=9):
    """Calcula el MACD, la línea de señal y el histograma."""
    ema_fast = calcular_ema(precios, fast_period)
    ema_slow = calcular_ema(precios, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = calcular_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calcular_atr(df_ohlcv, periodo=14):
    """Calcula el Average True Range (ATR)."""
    high_low = df_ohlcv['high'] - df_ohlcv['low']
    high_close = np.abs(df_ohlcv['high'] - df_ohlcv['close'].shift())
    low_close = np.abs(df_ohlcv['low'] - df_ohlcv['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=periodo).mean()
    return atr

def calcular_vwap(df_ohlcv):
    """Calcula el Volume Weighted Average Price (VWAP).
    Requiere 'close', 'volume' y 'high', 'low' para un cálculo más preciso.
    Aquí se usa una simplificación (TP * V) / V.
    """
    # Typical Price (TP) = (High + Low + Close) / 3
    df_ohlcv['tp'] = (df_ohlcv['high'] + df_ohlcv['low'] + df_ohlcv['close']) / 3
    df_ohlcv['tp_volume'] = df_ohlcv['tp'] * df_ohlcv['volume']
    
    # VWAP es una suma acumulativa, para un cálculo por vela, se puede simplificar
    # o calcular sobre un periodo. Para el filtro, usaremos el VWAP de la vela actual
    # o un VWAP de un periodo corto si se desea una línea.
    # Para el propósito de este filtro, asumiremos que necesitamos el VWAP de la vela actual
    # o un VWAP de un periodo reciente.
    # Una implementación más robusta de VWAP es acumulativa desde el inicio del día/sesión.
    # Para un filtro simple, podemos usar un VWAP de un periodo corto.
    
    # Simplificación: VWAP de la vela actual (TP) o un VWAP de 20 periodos
    vwap = df_ohlcv['tp_volume'].rolling(window=20).sum() / df_ohlcv['volume'].rolling(window=20).sum()
    return vwap

# --- 3. Función Principal de Identificación de Oportunidades ---

def identificar_oportunidades_trading_bitget(trading_pairs, timeframe='1h', limit=200):
    """
    Sistema de trading para Bitget con estrategia EMA Crossover + MACD + Volume Filter.
    """
    all_predictions = []

    for pair in trading_pairs:
        print(f"\nAnalizando {pair} en Bitget...")
        try:
            # Obtener datos OHLCV históricos
            # El 'limit' define cuántas velas históricas se obtienen.
            # Necesitamos suficientes para calcular todos los indicadores (EMA21, MACD, ATR14, VWAP20)
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(timezone) # Convertir a Bogotá

            if len(df) < max(26, 20, 14): # Asegurar suficientes datos para todos los indicadores
                print(f"  No hay suficientes datos históricos para {pair}. Saltando.")
                continue

            # Calcular Indicadores
            df['ema8'] = calcular_ema(df['close'], 8)
            df['ema21'] = calcular_ema(df['close'], 21)
            df['macd_line'], df['signal_line'], df['macd_hist'] = calcular_macd(df['close'], 12, 26, 9)
            df['atr'] = calcular_atr(df)
            df['vwap'] = calcular_vwap(df)
            
            # Calcular volumen promedio para el filtro
            df['volume_avg'] = df['volume'].rolling(window=20).mean() # Volumen promedio de 20 periodos

            # Eliminar filas con NaN resultantes de los cálculos de indicadores
            df.dropna(inplace=True)

            if df.empty:
                print(f"  DataFrame vacío después de calcular indicadores para {pair}. Saltando.")
                continue

            # Tomar la última vela (la más reciente) para la señal
            last_candle = df.iloc[-1]
            
            # --- Lógica de Señales (EMA Crossover + MACD + Volume + VWAP) ---
            señal_compra = False
            señal_venta = False
            confianza = 0
            nivel_oportunidad = "BAJA"
            
            precio_actual = last_candle['close']
            precio_entrada = precio_actual # Asumimos entrada al precio actual de la vela de señal

            # Condiciones para Entrada LONG
            cond_ema_long = last_candle['ema8'] > last_candle['ema21'] and df.iloc[-2]['ema8'] <= df.iloc[-2]['ema21'] # Cruce alcista
            cond_macd_long = last_candle['macd_line'] > last_candle['signal_line'] and last_candle['macd_hist'] > 0 # MACD alcista y histograma positivo
            cond_volume_long = last_candle['volume'] > (1.5 * last_candle['volume_avg']) # Volumen 1.5x promedio
            cond_vwap_long = last_candle['close'] > last_candle['vwap'] # Price Action por encima del VWAP

            if cond_ema_long and cond_macd_long and cond_volume_long and cond_vwap_long:
                señal_compra = True
                confianza = 85 # Alta confianza por cumplir todas las condiciones
                nivel_oportunidad = "MUY ALTA"
                # Ajustar confianza basada en la fuerza del MACD histograma
                confianza += min(10, last_candle['macd_hist'] / abs(last_candle['macd_line']) * 100 if last_candle['macd_line'] != 0 else 0)
                confianza = min(100, confianza)

            # Condiciones para Entrada SHORT (Inverso)
            cond_ema_short = last_candle['ema8'] < last_candle['ema21'] and df.iloc[-2]['ema8'] >= df.iloc[-2]['ema21'] # Cruce bajista
            cond_macd_short = last_candle['macd_line'] < last_candle['signal_line'] and last_candle['macd_hist'] < 0 # MACD bajista y histograma negativo
            cond_volume_short = last_candle['volume'] > (1.5 * last_candle['volume_avg']) # Volumen 1.5x promedio
            cond_vwap_short = last_candle['close'] < last_candle['vwap'] # Price Action por debajo del VWAP

            if cond_ema_short and cond_macd_short and cond_volume_short and cond_vwap_short:
                señal_venta = True
                confianza = 85 # Alta confianza
                nivel_oportunidad = "MUY ALTA"
                # Ajustar confianza basada en la fuerza del MACD histograma
                confianza += min(10, abs(last_candle['macd_hist']) / abs(last_candle['macd_line']) * 100 if last_candle['macd_line'] != 0 else 0)
                confianza = min(100, confianza)

            # --- Gestión de Riesgos (Stop Loss y Take Profit basados en ATR) ---
            stop_loss = None
            take_profit_1 = None
            take_profit_2 = None
            take_profit_3 = None
            ganancia_estimada = None
            duracion_estimada = None # Esto es más difícil de estimar con precisión en tiempo real

            atr_val = last_candle['atr']
            if np.isnan(atr_val): # Fallback si ATR no se pudo calcular (pocos datos)
                atr_val = precio_actual * 0.005 # 0.5% del precio como fallback

            # Factores de ATR para SL/TP (ajustables)
            factor_sl = 2.0 # 2 veces ATR para Stop Loss
            factor_tp1 = 1.5 # 1.5 veces ATR para TP1
            factor_tp2 = 2.5 # 2.5 veces ATR para TP2
            factor_tp3 = 4.0 # 4.0 veces ATR para TP3

            if señal_compra:
                stop_loss = precio_entrada - (atr_val * factor_sl)
                take_profit_1 = precio_entrada + (atr_val * factor_tp1)
                take_profit_2 = precio_entrada + (atr_val * factor_tp2)
                take_profit_3 = precio_entrada + (atr_val * factor_tp3)
                ganancia_estimada = (take_profit_2 / precio_entrada - 1) * 100 # Basado en TP2
                # Duración estimada es heurística, puede ser ajustada o eliminada
                duracion_estimada = max(1, int(atr_val * 100 / (last_candle['close'] * last_candle['volatility'] / 100)) if 'volatility' in last_candle else 4) # Heurística
                
            elif señal_venta:
                stop_loss = precio_entrada + (atr_val * factor_sl)
                take_profit_1 = precio_entrada - (atr_val * factor_tp1)
                take_profit_2 = precio_entrada - (atr_val * factor_tp2)
                take_profit_3 = precio_entrada - (atr_val * factor_tp3)
                ganancia_estimada = (1 - take_profit_2 / precio_entrada) * 100
                duracion_estimada = max(1, int(atr_val * 100 / (last_candle['close'] * last_candle['volatility'] / 100)) if 'volatility' in last_candle else 4) # Heurística

            # --- Filtro Adicional: No operar en resistencias/soportes fuertes ---
            # Esto es complejo de automatizar sin un módulo de detección de S/R.
            # Una simplificación es evitar operar si el precio está muy cerca de un máximo/mínimo reciente
            # o si el ATR es muy bajo (mercado lateral sin momentum).
            # Por ahora, la condición de VWAP y volumen ya actúan como un filtro de momentum.
            # Para S/R fuertes, se necesitaría un análisis de Price Action más profundo o niveles predefinidos.
            
            # Si la señal es muy débil o el mercado es muy lateral, podemos descartarla
            if confianza < 60: # Umbral de confianza para operar
                señal_compra = False
                señal_venta = False
                nivel_oportunidad = "BAJA"

            # --- Registro de la Predicción ---
            all_predictions.append({
                'par_trading': pair,
                'timestamp': last_candle['timestamp'],
                'hora_colombia': last_candle['timestamp'].strftime('%H:%M'),
                'dia_semana': last_candle['timestamp'].strftime('%a'), # Abreviatura del día
                'precio_actual': round(precio_actual, 8),
                'precio_entrada': round(precio_entrada, 8),
                'señal': 'COMPRA' if señal_compra else 'VENTA' if señal_venta else 'ESPERAR',
                'confianza': round(confianza, 1),
                'nivel_oportunidad': nivel_oportunidad,
                'stop_loss': round(stop_loss, 8) if stop_loss else None,
                'take_profit_1': round(take_profit_1, 8) if take_profit_1 else None,
                'take_profit_2': round(take_profit_2, 8) if take_profit_2 else None,
                'take_profit_3': round(take_profit_3, 8) if take_profit_3 else None,
                'duracion_horas': duracion_estimada,
                'momento_cierre': (last_candle['timestamp'] + timedelta(hours=duracion_estimada)) if duracion_estimada else None,
                'ganancia_estimada_%': round(ganancia_estimada, 2) if ganancia_estimada else None,
                'ema8': round(last_candle['ema8'], 8),
                'ema21': round(last_candle['ema21'], 8),
                'macd_line': round(last_candle['macd_line'], 8),
                'signal_line': round(last_candle['signal_line'], 8),
                'macd_hist': round(last_candle['macd_hist'], 8),
                'volume': round(last_candle['volume'], 2),
                'volume_avg': round(last_candle['volume_avg'], 2),
                'vwap': round(last_candle['vwap'], 8),
                'atr': round(atr_val, 8)
            })

            # --- Ejecución de Órdenes (¡Solo para producción y con testnet primero!) ---
            # Esta parte es crucial para un bot real.
            # Se recomienda usar un sistema de gestión de posiciones para evitar abrir múltiples operaciones.
            # if señal_compra:
            #     print(f"  >>> EJECUTANDO ORDEN DE COMPRA para {pair} a {precio_entrada}")
            #     try:
            #         # Ejemplo de orden de mercado. Ajusta el 'amount' (cantidad) según tu gestión de capital.
            #         # Asegúrate de que el 'symbol' sea el correcto para Bitget (e.g., 'BTC/USDT:USDT' para futuros)
            #         order = exchange.create_order(
            #             symbol=pair,
            #             type='market', # o 'limit' si quieres un precio específico
            #             side='buy',
            #             amount=0.001, # Cantidad de la criptomoneda a comprar (ejemplo)
            #             params={'stopLoss': {'price': stop_loss}, 'takeProfit': {'price': take_profit_2}} # Para futuros
            #         )
            #         print(f"    Orden de compra enviada: {order['id']}")
            #     except Exception as e:
            #         print(f"    ERROR al enviar orden de compra: {e}")
            # elif señal_venta:
            #     print(f"  >>> EJECUTANDO ORDEN DE VENTA (SHORT) para {pair} a {precio_entrada}")
            #     try:
            #         order = exchange.create_order(
            #             symbol=pair,
            #             type='market',
            #             side='sell',
            #             amount=0.001, # Cantidad a vender (ejemplo)
            #             params={'stopLoss': {'price': stop_loss}, 'takeProfit': {'price': take_profit_2}} # Para futuros
            #         )
            #         print(f"    Orden de venta enviada: {order['id']}")
            #     except Exception as e:
            #         print(f"    ERROR al enviar orden de venta: {e}")

        except ccxt.NetworkError as e:
            print(f"  [ERROR de Red] No se pudo conectar a Bitget para {pair}: {e}")
        except ccxt.ExchangeError as e:
            print(f"  [ERROR del Exchange] Problema con la API de Bitget para {pair}: {e}")
        except Exception as e:
            print(f"  [ERROR General] Ocurrió un error inesperado para {pair}: {e}")
            
    return pd.DataFrame(all_predictions)

# --- 4. Función para Mostrar Mejores Oportunidades ---

def mostrar_mejores_oportunidades_multi_asset(df, top_n=15):
    """Muestra las mejores oportunidades de trading para múltiples activos."""
    oportunidades = df[
        (df['señal'] != 'ESPERAR') & 
        (df['confianza'] >= 60) # Solo señales con confianza media o alta
    ].sort_values('confianza', ascending=False).head(top_n)
    
    print("\n🚀 MEJORES OPORTUNIDADES DE TRADING EN MÚLTIPLES PARES (Bitget)")
    print("=" * 80)
    
    if oportunidades.empty:
        print("No se encontraron oportunidades de trading con alta confianza en este momento.")
        return

    for idx, row in oportunidades.iterrows():
        print(f"\n📈 PAR: {row['par_trading']}")
        print(f"⏰ {row['dia_semana']} {row['hora_colombia']} - {row['timestamp'].strftime('%d/%m/%Y')}")
        print(f"📊 SEÑAL: {row['señal']} | Confianza: {row['confianza']}% | {row['nivel_oportunidad']}")
        print(f"💰 Precio Entrada: ${row['precio_entrada']:.8f}") # Formato para mayor precisión
        
        if row['señal'] != 'ESPERAR':
            print(f"🛑 Stop Loss: ${row['stop_loss']:.8f}")
            print(f"🎯 Take Profits: TP1: ${row['take_profit_1']:.8f} | TP2: ${row['take_profit_2']:.8f} | TP3: ${row['take_profit_3']:.8f}")
            if row['duracion_horas']:
                print(f"⏱️  Duración Estimada: {row['duracion_horas']}h | Cierre Estimado: {row['momento_cierre'].strftime('%H:%M %d/%m')}")
            if row['ganancia_estimada_%']:
                print(f"📈 Ganancia Estimada: {row['ganancia_estimada_%.2f']}%")
        
        print(f"📊 EMA8: {row['ema8']:.8f} | EMA21: {row['ema21']:.8f}")
        print(f"📊 MACD: {row['macd_line']:.8f} | Signal: {row['signal_line']:.8f} | Hist: {row['macd_hist']:.8f}")
        print(f"📊 Volumen: {row['volume']:.2f} (Avg: {row['volume_avg']:.2f}) | VWAP: {row['vwap']:.8f} | ATR: {row['atr']:.8f}")
        print("-" * 60)

# --- 5. Ejecución Principal ---

# Lista de pares de trading a analizar (asegúrate que Bitget los soporte en el tipo de mercado elegido)
# Para futuros, a menudo son 'BTCUSDT', 'ETHUSDT', etc. sin la barra.
# Verifica la documentación de Bitget o usa exchange.load_markets() para ver los símbolos exactos.
trading_pairs_list = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SHIB/USDT', 
    # 'LPT/USDC', 'RVN/USDC', # Estos pueden ser pares de spot o tener símbolos diferentes
    'WLD/USDT', 'NEAR/USDT', 'TRX/USDT', 'STX/USDT',
    'NEO/USDT', 'COS/USDT', 'PYR/USDT', 'AVA/USDT'
]

# Ajusta el timeframe según tu estrategia (e.g., '1h', '4h', '1d')
TIMEFRAME = '1h' 
# Ajusta el límite de velas históricas. Necesitas suficientes para los cálculos de indicadores.
# Por ejemplo, para EMA21 y MACD26, necesitas al menos 26 velas. Para ATR14, 14 velas.
# Un límite de 100-200 es generalmente seguro.
CANDLE_LIMIT = 200 

print(f"Generando análisis avanzado de trading para múltiples criptomonedas en Bitget ({TIMEFRAME} timeframe)...")
df_multi_trading = identificar_oportunidades_trading_bitget(trading_pairs_list, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)

# Mostrar mejores oportunidades
mostrar_mejores_oportunidades_multi_asset(df_multi_trading, 20)

# Estadísticas generales
total_señales_global = len(df_multi_trading[df_multi_trading['señal'] != 'ESPERAR'])
señales_alta_confianza_global = len(df_multi_trading[df_multi_trading['confianza'] >= 70])

print(f"\n📊 RESUMEN ESTADÍSTICO GLOBAL:")
print(f"Total de señales detectadas en todos los pares: {total_señales_global}")
print(f"Señales de alta confianza (≥70%) en todos los pares: {señales_alta_confianza_global}")
if total_señales_global > 0:
    print(f"Promedio de confianza de señales activas: {df_multi_trading[df_multi_trading['confianza'] > 0]['confianza'].mean():.1f}%")
else:
    print("No hay señales activas para calcular el promedio de confianza.")

# Exportar a CSV para análisis adicional
df_multi_trading.to_csv('predicciones_multi_crypto_trading_bitget.csv', index=False, sep = ';', decimal = ',')
print(f"\n💾 Datos exportados a 'predicciones_multi_crypto_trading_bitget.csv'")

