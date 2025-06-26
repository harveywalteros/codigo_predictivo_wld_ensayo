import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

def generar_predicciones_ejemplo():
    """
    Genera un ejemplo de predicciones basado en patrones típicos de WLD/USDT
    """
    # Configurar timezone de Bogotá
    timezone = pytz.timezone('America/Bogota')
    
    # Precio base actual estimado (basado en búsquedas web)
    precio_base = 2.40  # Precio aproximado actual de WLD/USDT
    
    # Generar timestamp inicial
    inicio = datetime.now(timezone)
    inicio = inicio.replace(minute=0, second=0, microsecond=0)  # Redondear a la hora
    
    predicciones = []
    
    # Patrones típicos por hora del día (basado en análisis de mercado crypto)
    patrones_horarios = {
        0: {'volatilidad': 2.1, 'tendencia': -0.2},   # Medianoche - baja actividad
        1: {'volatilidad': 1.8, 'tendencia': -0.3},   # Madrugada
        2: {'volatilidad': 1.5, 'tendencia': -0.1},   # Muy baja actividad
        3: {'volatilidad': 1.7, 'tendencia': 0.1},    # Pre-apertura Asia
        4: {'volatilidad': 2.3, 'tendencia': 0.4},    # Apertura Asia
        5: {'volatilidad': 2.8, 'tendencia': 0.6},    # Asia activa
        6: {'volatilidad': 3.2, 'tendencia': 0.3},    # Asia media mañana
        7: {'volatilidad': 3.0, 'tendencia': 0.2},    # Pre-Europa
        8: {'volatilidad': 4.1, 'tendencia': 0.8},    # Apertura Europa - ALTA OPORTUNIDAD
        9: {'volatilidad': 4.5, 'tendencia': 1.2},    # Europa activa - ALTA OPORTUNIDAD
        10: {'volatilidad': 4.2, 'tendencia': 0.9},   # Europa mañana - ALTA OPORTUNIDAD
        11: {'volatilidad': 3.8, 'tendencia': 0.5},   # Europa media mañana
        12: {'volatilidad': 3.5, 'tendencia': 0.3},   # Mediodía Europa
        13: {'volatilidad': 3.9, 'tendencia': 0.7},   # Tarde Europa
        14: {'volatilidad': 4.3, 'tendencia': 1.1},   # Pre-USA - ALTA OPORTUNIDAD
        15: {'volatilidad': 4.8, 'tendencia': 1.4},   # Apertura USA - MUY ALTA OPORTUNIDAD
        16: {'volatilidad': 4.6, 'tendencia': 1.2},   # USA activa - ALTA OPORTUNIDAD
        17: {'volatilidad': 4.2, 'tendencia': 0.8},   # USA tarde
        18: {'volatilidad': 3.9, 'tendencia': 0.6},   # USA final tarde
        19: {'volatilidad': 3.5, 'tendencia': 0.4},   # USA noche
        20: {'volatilidad': 3.1, 'tendencia': 0.2},   # USA noche tardía
        21: {'volatilidad': 2.8, 'tendencia': 0.0},   # Cierre USA
        22: {'volatilidad': 2.4, 'tendencia': -0.1},  # Post-cierre
        23: {'volatilidad': 2.2, 'tendencia': -0.2}   # Pre-medianoche
    }
    
    # Generar predicciones para 48 horas
    for hora in range(48):
        timestamp_futuro = inicio + timedelta(hours=hora + 1)
        hora_del_dia = timestamp_futuro.hour
        dia_semana = timestamp_futuro.weekday()
        
        # Obtener patrón base para esa hora
        patron = patrones_horarios[hora_del_dia]
        volatilidad_base = patron['volatilidad']
        tendencia_base = patron['tendencia']
        
        # Ajustar por día de la semana
        factor_dia = 1.0
        if dia_semana in [1, 2, 3]:  # Martes, miércoles, jueves - días más activos
            factor_dia = 1.15
        elif dia_semana in [5, 6]:  # Sábado, domingo - menos activos
            factor_dia = 0.85
        elif dia_semana == 0:  # Lunes - actividad media-alta
            factor_dia = 1.05
        elif dia_semana == 4:  # Viernes - actividad media
            factor_dia = 1.10
        
        # Calcular volatilidad y tendencia ajustadas
        volatilidad_ajustada = volatilidad_base * factor_dia
        tendencia_ajustada = tendencia_base * factor_dia
        
        # Simular precio futuro
        precio_futuro = precio_base + np.random.randn() * volatilidad_ajustada + tendencia_ajustada
        
        # Guardar predicción
        predicciones.append({
            'timestamp': timestamp_futuro,
            'precio': precio_futuro,
            'volatilidad': volatilidad_ajustada,
            'tendencia': tendencia_ajustada
        })
    
    # Convertir a DataFrame de pandas
    df_predicciones = pd.DataFrame(predicciones)
    return df_predicciones

# Ejemplo de uso
df_predicciones = generar_predicciones_ejemplo()
print(df_predicciones)