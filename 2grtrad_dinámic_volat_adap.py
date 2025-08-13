import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
import ccxt # Importar la librería ccxt para interactuar con Bitget
import time
import os # Para variables de entorno

warnings.filterwarnings('ignore')

# --- 1. Configuración y Credenciales (¡IMPORTANTE: Usar variables de entorno!) ---
# Se recomienda encarecidamente NO poner las claves directamente aquí.
# Ejemplo: export BITGET_API_KEY='tu_api_key'
#          export BITGET_SECRET='tu_secret'
#          export BITGET_PASSWORD='tu_password'

BITGET_API_KEY = os.getenv('BITGET_API_KEY')
BITGET_SECRET = os.getenv('BITGET_SECRET')
BITGET_PASSWORD = os.getenv('BITGET_PASSWORD') # Si Bitget requiere una contraseña de trading

if not all([BITGET_API_KEY, BITGET_SECRET, BITGET_PASSWORD]):
    print("ADVERTENCIA: Las credenciales de Bitget no están configuradas en las variables de entorno.")
    print("El bot funcionará en modo de simulación de órdenes. Para trading real, configura BITGET_API_KEY, BITGET_SECRET y BITGET_PASSWORD.")
    LIVE_TRADING_ENABLED = False
else:
    LIVE_TRADING_ENABLED = True

# --- 2. Inicialización del Exchange ---
exchange = ccxt.bitget({
    'apiKey': BITGET_API_KEY,
    'secret': BITGET_SECRET,
    'password': BITGET_PASSWORD, # Opcional, si Bitget lo requiere
    'options': {
        'defaultType': 'swap', # O 'spot' si operas en el mercado spot
    },
    'enableRateLimit': True, # Para evitar exceder los límites de la API
})

# --- 3. Funciones de Indicadores Técnicos (Reutilizadas y Mejoradas) ---

def calcular_atr(df_ohlcv, periodo=14):
    """
    Calcula el Average True Range (ATR) a partir de un DataFrame OHLCV.
    Requiere columnas 'high', 'low', 'close'.
    """
    if df_ohlcv.empty or len(df_ohlcv) < periodo:
        return None # No hay suficientes datos para calcular ATR

    high_low = df_ohlcv['high'] - df_ohlcv['low']
    high_close = np.abs(df_ohlcv['high'] - df_ohlcv['close'].shift())
    low_close = np.abs(df_ohlcv['low'] - df_ohlcv['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=periodo).mean()
    return atr.iloc[-1] # Devolver el último valor de ATR

# --- 4. Funciones de Interacción con Bitget ---

def fetch_ohlcv(symbol, timeframe='1h', limit=100):
    """
    Obtiene datos OHLCV de Bitget para un símbolo y marco de tiempo dados.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ccxt.NetworkError as e:
        print(f"Error de red al obtener OHLCV para {symbol}: {e}")
        return pd.DataFrame()
    except ccxt.ExchangeError as e:
        print(f"Error del exchange al obtener OHLCV para {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error inesperado al obtener OHLCV para {symbol}: {e}")
        return pd.DataFrame()

def get_balance(currency):
    """
    Obtiene el balance disponible de una moneda.
    """
    try:
        balance = exchange.fetch_balance()
        if currency in balance['free']:
            return balance['free'][currency]
        return 0.0
    except ccxt.NetworkError as e:
        print(f"Error de red al obtener balance de {currency}: {e}")
        return 0.0
    except ccxt.ExchangeError as e:
        print(f"Error del exchange al obtener balance de {currency}: {e}")
        return 0.0
    except Exception as e:
        print(f"Error inesperado al obtener balance de {currency}: {e}")
        return 0.0

def place_limit_order(symbol, side, amount, price):
    """
    Coloca una orden límite en Bitget.
    """
    if not LIVE_TRADING_ENABLED:
        print(f"[SIMULACIÓN] Orden {side} {amount:.8f} {symbol} @ {price:.8f}")
        return {'info': 'Simulated order'}

    try:
        order = exchange.create_order(symbol, 'limit', side, amount, price)
        print(f"Orden {side} {amount:.8f} {symbol} @ {price:.8f} colocada. ID: {order['id']}")
        return order
    except ccxt.InsufficientFunds as e:
        print(f"Fondos insuficientes para orden {side} {symbol}: {e}")
        return None
    except ccxt.InvalidOrder as e:
        print(f"Orden inválida {side} {symbol}: {e}")
        return None
    except ccxt.NetworkError as e:
        print(f"Error de red al colocar orden {side} {symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"Error del exchange al colocar orden {side} {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado al colocar orden {side} {symbol}: {e}")
        return None

def cancel_all_open_orders(symbol):
    """
    Cancela todas las órdenes abiertas para un símbolo.
    """
    if not LIVE_TRADING_ENABLED:
        print(f"[SIMULACIÓN] Cancelando todas las órdenes abiertas para {symbol}")
        return True

    try:
        orders = exchange.fetch_open_orders(symbol)
        if not orders:
            return True
        for order in orders:
            exchange.cancel_order(order['id'], symbol)
            print(f"Orden {order['id']} de {symbol} cancelada.")
        return True
    except ccxt.NetworkError as e:
        print(f"Error de red al cancelar órdenes para {symbol}: {e}")
        return False
    except ccxt.ExchangeError as e:
        print(f"Error del exchange al cancelar órdenes para {symbol}: {e}")
        return False
    except Exception as e:
        print(f"Error inesperado al cancelar órdenes para {symbol}: {e}")
        return False

# --- 5. Lógica de la Estrategia de Grid Trading Dinámico ---

def calculate_grid_levels(current_price, atr_value, num_levels, atr_multiplier):
    """
    Calcula los niveles de la grid de compra y venta.
    """
    if atr_value is None or atr_value <= 0:
        # Fallback si ATR no es válido, usar un porcentaje fijo o un valor mínimo
        print("ATR no válido, usando espaciado fijo del 0.5% del precio actual.")
        spacing = current_price * 0.005
    else:
        spacing = atr_value * atr_multiplier

    buy_levels = []
    sell_levels = []

    # Generar niveles de compra por debajo del precio actual
    for i in range(1, num_levels + 1):
        buy_level = current_price - (spacing * i)
        buy_levels.append(buy_level)
    buy_levels.sort(reverse=True) # De mayor a menor precio

    # Generar niveles de venta por encima del precio actual
    for i in range(1, num_levels + 1):
        sell_level = current_price + (spacing * i)
        sell_levels.append(sell_level)
    sell_levels.sort() # De menor a mayor precio

    return buy_levels, sell_levels, spacing

def manage_dynamic_grid(symbol, usdt_balance, total_capital_usdt, grid_params):
    """
    Gestiona la estrategia de Grid Trading Dinámico para un par.
    """
    print(f"\n--- Gestionando Grid para {symbol} ---")

    # 5.1 Obtener datos OHLCV y precio actual
    ohlcv_df = fetch_ohlcv(symbol, timeframe=grid_params['timeframe'], limit=grid_params['atr_period'] + 50)
    if ohlcv_df.empty:
        print(f"No se pudieron obtener datos OHLCV para {symbol}. Saltando.")
        return

    current_price = ohlcv_df['close'].iloc[-1]
    print(f"Precio actual de {symbol}: {current_price:.8f}")

    # 5.2 Calcular ATR
    atr_value = calcular_atr(ohlcv_df, periodo=grid_params['atr_period'])
    print(f"ATR({grid_params['atr_period']}): {atr_value:.8f}")

    # 5.3 Calcular niveles de la grid
    buy_levels, sell_levels, actual_spacing = calculate_grid_levels(
        current_price, atr_value, grid_params['num_levels'], grid_params['atr_multiplier']
    )
    print(f"Espaciado de Grid (ATR * {grid_params['atr_multiplier']}): {actual_spacing:.8f}")
    print(f"Niveles de Compra: {[f'{lvl:.8f}' for lvl in buy_levels]}")
    print(f"Niveles de Venta: {[f'{lvl:.8f}' for lvl in sell_levels]}")

    # 5.4 Calcular capital por nivel
    capital_per_level_usdt = total_capital_usdt * (grid_params['capital_per_level_pct'] / 100)
    print(f"Capital por nivel: {capital_per_level_usdt:.2f} USDT")

    # 5.5 Cancelar órdenes existentes para reajustar la grid
    print(f"Cancelando órdenes antiguas para {symbol}...")
    cancel_all_open_orders(symbol)
    time.sleep(1) # Pequeña pausa para que el exchange procese las cancelaciones

    # 5.6 Colocar nuevas órdenes de la grid
    placed_orders_count = 0

    # Colocar órdenes de compra
    for level in buy_levels:
        # Asegurarse de que el precio de la orden sea menor que el precio actual para órdenes de compra
        if level < current_price:
            amount_to_buy = capital_per_level_usdt / level
            # Redondear la cantidad a la precisión del exchange si es necesario
            # (Esto es crucial en trading real, Bitget tiene 'min_qty', 'qty_precision', etc.)
            # Por simplicidad, aquí solo un redondeo básico
            amount_to_buy = round(amount_to_buy, 6) # Ajustar según la precisión del par
            if amount_to_buy * level > 10: # Ejemplo: Mínimo de 10 USDT por orden
                order = place_limit_order(symbol, 'buy', amount_to_buy, level)
                if order:
                    placed_orders_count += 1
            else:
                print(f"Cantidad de compra {amount_to_buy:.8f} {symbol} en {level:.8f} es muy pequeña. Saltando.")

    # Colocar órdenes de venta
    for level in sell_levels:
        # Asegurarse de que el precio de la orden sea mayor que el precio actual para órdenes de venta
        if level > current_price:
            # Para órdenes de venta, necesitamos tener la criptomoneda base (ej. BTC en BTC/USDT)
            # Aquí asumimos que ya tenemos algo de la criptomoneda base para vender.
            # En un bot real, esto requeriría un seguimiento de la posición.
            amount_to_sell = capital_per_level_usdt / current_price # Estimación de la cantidad a vender
            amount_to_sell = round(amount_to_sell, 6) # Ajustar según la precisión del par
            if amount_to_sell * level > 10: # Ejemplo: Mínimo de 10 USDT por orden
                order = place_limit_order(symbol, 'sell', amount_to_sell, level)
                if order:
                    placed_orders_count += 1
            else:
                print(f"Cantidad de venta {amount_to_sell:.8f} {symbol} en {level:.8f} es muy pequeña. Saltando.")

    print(f"Total de {placed_orders_count} órdenes de grid colocadas para {symbol}.")

# --- 6. Bucle Principal de Ejecución ---

def run_grid_bot(trading_pairs, total_capital_usdt, refresh_interval_hours=4):
    """
    Función principal para ejecutar el bot de grid trading.
    """
    print("\n🚀 Iniciando Bot de Grid Trading Dinámico 🚀")
    print(f"Pares a operar: {trading_pairs}")
    print(f"Capital total asignado: {total_capital_usdt:.2f} USDT")
    print(f"Intervalo de rebalanceo: {refresh_interval_hours} horas")
    print(f"Modo de trading: {'REAL' if LIVE_TRADING_ENABLED else 'SIMULACIÓN'}")

    # Parámetros de la grid (pueden ser específicos por par si se desea)
    grid_parameters = {
        'timeframe': '1h',
        'atr_period': 14,
        'num_levels': 5, # 5 niveles de compra y 5 de venta = 10 órdenes en total
        'atr_multiplier': 1.2, # Espaciado = ATR * 1.2
        'capital_per_level_pct': 6 # 6% del capital total por nivel (5-7% recomendado)
    }

    last_refresh_time = datetime.now() - timedelta(hours=refresh_interval_hours + 1) # Forzar primera ejecución

    while True:
        current_time = datetime.now()
        if (current_time - last_refresh_time).total_seconds() / 3600 >= refresh_interval_hours:
            print(f"\n--- Rebalanceando Grids a las {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
            last_refresh_time = current_time

            # Obtener balance de USDT (asumiendo que es la moneda base para el capital)
            usdt_balance = get_balance('USDT')
            print(f"Balance disponible de USDT: {usdt_balance:.2f}")

            if usdt_balance < total_capital_usdt * 0.1 and LIVE_TRADING_ENABLED: # Si el balance es muy bajo
                print("ADVERTENCIA: Balance de USDT bajo. Considera recargar o ajustar el capital total.")
                # Podrías pausar el bot o reducir el capital_per_level_pct aquí

            for pair in trading_pairs:
                manage_dynamic_grid(pair, usdt_balance, total_capital_usdt, grid_parameters)
                time.sleep(5) # Pequeña pausa entre pares para evitar límites de API

        print(f"\nEsperando {refresh_interval_hours} horas para el próximo rebalanceo... (Último: {last_refresh_time.strftime('%H:%M')})")
        time.sleep(3600) # Esperar 1 hora antes de revisar de nuevo (o un intervalo más corto si se desea más reactividad)

# --- 7. Ejecución Principal ---
if __name__ == "__main__":
    # Lista de pares de trading a analizar (asegúrate que sean válidos en Bitget)
    # Para Bitget, los símbolos de futuros perpetuos suelen ser 'BTC/USDT:USDT' o 'BTCUSDT'
    # Para spot, 'BTC/USDT'
    # Verifica la documentación de Bitget o usa exchange.load_markets() para ver los símbolos exactos.
    
    # Ejemplo para futuros perpetuos en Bitget (comunes para grid trading)
    trading_pairs_list = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT',
        'XRP/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT',
        'NEAR/USDT:USDT', 'TRX/USDT:USDT', 'STX/USDT:USDT', 'WLD/USDT:USDT'
    ]
    
    # Si quieres operar en SPOT, cambia los símbolos a 'BTC/USDT', 'ETH/USDT', etc.
    # y ajusta 'defaultType': 'spot' en la inicialización del exchange.

    # Capital total que el bot puede usar para todas las grids (en USDT)
    # ¡AJUSTA ESTO SEGÚN TU CAPITAL REAL Y TOLERANCIA AL RIESGO!
    total_capital_for_bot = 1000 # Ejemplo: 1000 USDT

    run_grid_bot(trading_pairs_list, total_capital_for_bot, refresh_interval_hours=4)

