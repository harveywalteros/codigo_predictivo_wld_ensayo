import pandas as pd
import numpy as np
import talib
import ccxt
import websocket

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("TA-Lib version:", talib.__version__)
print("CCXT version:", ccxt.__version__)
print("WebSocket-client disponible:", hasattr(websocket, 'WebSocketApp'))