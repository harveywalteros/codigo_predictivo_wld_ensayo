import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
import ccxt
import time
import os # Para cargar variables de entorno

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DEL BOT ---
# ¡IMPORTANTE! Configura tus credenciales de Bitget y el par de trading
# Se recomienda usar variables de entorno o un archivo .env para mayor seguridad
# Ejemplo: export BITGET_API_KEY='TU_API_KEY'
#          export BITGET_API_SECRET='TU_API_SECRET'
BITGET_API_KEY = os.getenv('BITGET_API_KEY', 'TU_API_KEY_AQUI') # Reemplaza con tu API Key
BITGET_API_SECRET = os.getenv('BITGET_API_SECRET', 'TU_API_SECRET_AQUI') # Reemplaza con tu API Secret

SYMBOL = 'BTC/USDT' # Par de trading (ej. 'BTC/USDT', 'ETH/USDT', 'SOL/USDT')
TIMEFRAME = '15m'   # Timeframe de las velas (ej. '5m', '15m', '1h')
LOOKBACK_CANDLES = 200 # Número de velas históricas a cargar para cálculos

# Parámetros de la estrategia
BB_PERIOD = 20
BB_DEV = 2.0
RSI_PERIOD = 14
STOCH_RSI_K_PERIOD = 14
STOCH_RSI_D_PERIOD = 3
STOCH_RSI_SMOOTHING = 3

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
STOCH_RSI_OVERBOUGHT = 80
STOCH_RSI_OVERSOLD = 20

# Gestión de Riesgos (porcentajes)
STOP_LOSS_PERCENT = 0.015 # 1.5%
TAKE_PROFIT_PERCENT_1 = 0.020 # 2.0%
TAKE_PROFIT_PERCENT_2 = 0.030 # 3.0%

# --- FUNCIONES DE INDICADORES TÉCNICOS ---

def calculate_bollinger_bands(df, window=BB_PERIOD, num_std_dev=BB_DEV):
    """Calcula las Bandas de Bollinger."""
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['Upper_BB'] = df['SMA'] + (df['STD'] * num_std_dev)
    df['Lower_BB'] = df['SMA'] - (df['STD'] * num_std_dev)
    return df

def calculate_rsi(df, window=RSI_PERIOD):
    """Calcula el Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_stoch_rsi(df, k_period=STOCH_RSI_K_PERIOD, d_period=STOCH_RSI_D_PERIOD, smoothing=STOCH_RSI_SMOOTHING):
    """Calcula el Stochastic RSI."""
    # Calcular RSI primero
    df = calculate_rsi(df, window=k_period) # Usar k_period para el RSI interno del StochRSI

    # Calcular Stoch RSI
    lowest_rsi = df['RSI'].rolling(window=k_period).min()
    highest_rsi = df['RSI'].rolling(window=k_period).max()
    df['Stoch_RSI_K'] = ((df['RSI'] - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100
    df['Stoch_RSI_K'] = df['Stoch_RSI_K'].rolling(window=smoothing).mean() # Suavizado
    df['Stoch_RSI_D'] = df['Stoch_RSI_K'].rolling(window=d_period).mean() # Media móvil de %K

    return df

# --- CONEXIÓN Y OBTENCIÓN DE DATOS ---

def get_bitget_client():
    """Inicializa el cliente de Bitget."""
    try:
        exchange = ccxt.bitget({
            'apiKey': BITGET_API_KEY,
            'secret': BITGET_API_SECRET,
            'options': {
                'defaultType': 'spot', # O 'future' si operas futuros
            },
            'enableRateLimit': True, # Para evitar exceder límites de la API
        })
        # Cargar mercados para obtener información del símbolo
        exchange.load_markets()
        print(f"Conectado a Bitget. Servidor: {exchange.fetch_time()}")
        return exchange
    except Exception as e:
        print(f"Error al conectar con Bitget: {e}")
        return None

def fetch_ohlcv(exchange, symbol, timeframe, limit):
    """Obtiene datos OHLCV del exchange."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error al obtener datos OHLCV para {symbol}: {e}")
        return pd.DataFrame()

# --- LÓGICA DE TRADING ---

def analyze_and_signal(df):
    """
    Analiza el DataFrame con indicadores y genera señales de trading.
    Retorna la última señal y los niveles de precio.
    """
    if df.empty or len(df) < max(BB_PERIOD, RSI_PERIOD, STOCH_RSI_K_PERIOD, STOCH_RSI_D_PERIOD, STOCH_RSI_SMOOTHING):
        return None, None, None, None, None, None, None

    # Asegurarse de que los indicadores estén calculados
    df = calculate_bollinger_bands(df.copy())
    df = calculate_stoch_rsi(df.copy()) # Stoch RSI ya calcula RSI internamente

    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2] # Para el cruce de Stoch RSI

    current_price = last_candle['close']
    signal = "ESPERAR"
    confidence = 0
    entry_price = current_price
    stop_loss = None
    take_profit_1 = None
    take_profit_2 = None
    estimated_profit_percent = None

    # --- Lógica de Entrada (COMPRA - LONG) ---
    # Precio toca/cruza banda inferior
    price_at_lower_bb = last_candle['low'] <= last_candle['Lower_BB']

    # RSI sobreventa
    rsi_oversold = last_candle['RSI'] < RSI_OVERSOLD

    # Stoch RSI cruce alcista desde sobreventa
    stoch_rsi_cross_up = (prev_candle['Stoch_RSI_K'] < prev_candle['Stoch_RSI_D'] and
                          last_candle['Stoch_RSI_K'] > last_candle['Stoch_RSI_D'] and
                          last_candle['Stoch_RSI_K'] < STOCH_RSI_OVERSOLD) # Asegurarse que el cruce ocurre en zona de sobreventa

    if price_at_lower_bb and rsi_oversold and stoch_rsi_cross_up:
        signal = "COMPRA"
        entry_price = current_price
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
        take_profit_1 = entry_price * (1 + TAKE_PROFIT_PERCENT_1)
        take_profit_2 = entry_price * (1 + TAKE_PROFIT_PERCENT_2)
        estimated_profit_percent = TAKE_PROFIT_PERCENT_2 * 100 # Basado en TP2

        # Calcular confianza (ejemplo, puedes refinar esto)
        confidence = 60 # Base
        if last_candle['RSI'] < 20: confidence += 10
        if last_candle['Stoch_RSI_K'] < 10: confidence += 10
        if last_candle['close'] < last_candle['Lower_BB']: confidence += 5 # Más fuerte si cierra por debajo
        confidence = min(95, confidence) # Limitar confianza máxima

    # --- Lógica de Salida (Cierre de LONG - VENTA) ---
    # Esta lógica se aplicaría a una posición abierta.
    # Para este script, la usaremos para simular el TP/SL si se hubiera entrado.
    # En un bot real, tendrías que monitorear tu posición activa.

    # Salida por Take Profit (reversión a la media o RSI sobrecompra)
    # Si tuvieras una posición LONG abierta:
    # if current_price >= last_candle['SMA'] or last_candle['RSI'] > RSI_OVERBOUGHT:
    #     signal = "CERRAR_LONG_TP"

    # Salida por Stop Loss
    # if current_price <= stop_loss_de_posicion_abierta:
    #     signal = "CERRAR_LONG_SL"

    return signal, entry_price, stop_loss, take_profit_1, take_profit_2, confidence, estimated_profit_percent

def run_trading_bot():
    """Función principal para ejecutar el bot de trading."""
    exchange = get_bitget_client()
    if not exchange:
        return

    print(f"\nIniciando bot de trading para {SYMBOL} en timeframe {TIMEFRAME}...")
    print("Estrategia: Mean Reversion con Bollinger Bands + RSI Estocástico")
    print("-----------------------------------------------------------------")

    # Simulación de una posición abierta (para fines de demostración)
    # En un bot real, esto se manejaría con un estado persistente
    # current_position = {'status': 'none', 'entry_price': 0, 'signal_time': None}

    while True:
        df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, LOOKBACK_CANDLES)
        if df.empty:
            print("No se pudieron obtener datos. Reintentando en 60 segundos...")
            time.sleep(60)
            continue

        # Asegurarse de que el DataFrame tiene suficientes datos para los cálculos
        if len(df) < max(BB_PERIOD, RSI_PERIOD, STOCH_RSI_K_PERIOD, STOCH_RSI_D_PERIOD, STOCH_RSI_SMOOTHING):
            print(f"No hay suficientes datos ({len(df)} velas) para calcular todos los indicadores. Necesitas al menos {max(BB_PERIOD, RSI_PERIOD, STOCH_RSI_K_PERIOD, STOCH_RSI_D_PERIOD, STOCH_RSI_SMOOTHING)} velas. Esperando...")
            time.sleep(60)
            continue

        # Calcular todos los indicadores para el análisis
        df_indicators = calculate_bollinger_bands(df.copy())
        df_indicators = calculate_stoch_rsi(df_indicators.copy())

        # Obtener la última vela y sus indicadores
        last_candle = df_indicators.iloc[-1]
        current_price = last_candle['close']

        # Generar señal
        signal, entry_price, stop_loss, tp1, tp2, confidence, estimated_profit = \
            analyze_and_signal(df_indicators)

        print(f"\n[{datetime.now(pytz.timezone('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S')}]")
        print(f"Par: {SYMBOL} | Precio Actual: ${current_price:.8f}")
        print(f"RSI: {last_candle['RSI']:.2f} | Stoch RSI %K: {last_candle['Stoch_RSI_K']:.2f} | Stoch RSI %D: {last_candle['Stoch_RSI_D']:.2f}")
        print(f"BB Inferior: {last_candle['Lower_BB']:.8f} | BB Media: {last_candle['SMA']:.8f} | BB Superior: {last_candle['Upper_BB']:.8f}")

        if signal == "COMPRA":
            print(f"🚀 SEÑAL DE COMPRA DETECTADA para {SYMBOL}!")
            print(f"   Confianza: {confidence:.1f}%")
            print(f"   Precio de Entrada Sugerido: ${entry_price:.8f}")
            print(f"   Stop Loss Sugerido: ${stop_loss:.8f} ({STOP_LOSS_PERCENT*100:.1f}%)")
            print(f"   Take Profit 1 Sugerido: ${tp1:.8f} ({TAKE_PROFIT_PERCENT_1*100:.1f}%)")
            print(f"   Take Profit 2 Sugerido: ${tp2:.8f} ({TAKE_PROFIT_PERCENT_2*100:.1f}%)")
            print(f"   Ganancia Estimada: {estimated_profit:.1f}%")
            # Aquí iría la lógica para ejecutar la orden de compra real en Bitget
            # Ejemplo (descomentar para usar en un bot real, con manejo de errores y saldos):
            # try:
            #     order = exchange.create_market_buy_order(SYMBOL, amount_to_buy)
            #     print(f"Orden de COMPRA ejecutada: {order}")
            #     current_position = {'status': 'long', 'entry_price': entry_price, 'signal_time': datetime.now()}
            # except Exception as e:
            #     print(f"Error al ejecutar orden de compra: {e}")
        else:
            print(f"   Señal: {signal} (Esperando oportunidad...)")

        # En un bot real, aquí se añadiría la lógica para monitorear posiciones abiertas
        # y ejecutar órdenes de Take Profit o Stop Loss.

        # Esperar al siguiente intervalo de tiempo (ej. 15 minutos para un timeframe de 15m)
        # Ajusta el tiempo de espera según tu timeframe y la frecuencia de ejecución deseada
        sleep_time = 60 * int(TIMEFRAME.replace('m', '').replace('h', '')) # Convertir '15m' a 900 segundos
        print(f"Esperando {sleep_time / 60:.0f} minutos para la próxima verificación...")
        time.sleep(sleep_time)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    run_trading_bot()

