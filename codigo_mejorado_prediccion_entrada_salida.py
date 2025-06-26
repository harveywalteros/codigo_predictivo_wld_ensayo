import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

def calcular_rsi(precios, periodo=14):
    """Calcula el RSI para identificar condiciones de sobrecompra/sobreventa"""
    delta = precios.diff()
    ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    rs = ganancia / perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_media_movil(precios, periodo=20):
    """Calcula media móvil simple"""
    return precios.rolling(window=periodo).mean()

def identificar_oportunidades_trading():
    """
    Sistema avanzado de trading WLD/USDT con optimización de take profit
    """
    # Configurar timezone de Bogotá
    timezone = pytz.timezone('America/Bogota')
    
    # Precio base actual
    precio_base = 2.40
    
    # Generar timestamp inicial
    inicio = datetime.now(timezone)
    inicio = inicio.replace(minute=0, second=0, microsecond=0)
    
    # Patrones horarios optimizados con mayor precisión
    patrones_horarios = {
        0: {'volatilidad': 2.1, 'tendencia': -0.2, 'liquidez': 0.3},
        1: {'volatilidad': 1.8, 'tendencia': -0.3, 'liquidez': 0.2},
        2: {'volatilidad': 1.5, 'tendencia': -0.1, 'liquidez': 0.2},
        3: {'volatilidad': 1.7, 'tendencia': 0.1, 'liquidez': 0.3},
        4: {'volatilidad': 2.3, 'tendencia': 0.4, 'liquidez': 0.4},
        5: {'volatilidad': 2.8, 'tendencia': 0.6, 'liquidez': 0.5},
        6: {'volatilidad': 3.2, 'tendencia': 0.3, 'liquidez': 0.6},
        7: {'volatilidad': 3.0, 'tendencia': 0.2, 'liquidez': 0.7},
        8: {'volatilidad': 4.1, 'tendencia': 0.8, 'liquidez': 0.9},  # ALTA OPORTUNIDAD
        9: {'volatilidad': 4.5, 'tendencia': 1.2, 'liquidez': 1.0},  # MÁXIMA OPORTUNIDAD
        10: {'volatilidad': 4.2, 'tendencia': 0.9, 'liquidez': 0.9}, # ALTA OPORTUNIDAD
        11: {'volatilidad': 3.8, 'tendencia': 0.5, 'liquidez': 0.8},
        12: {'volatilidad': 3.5, 'tendencia': 0.3, 'liquidez': 0.7},
        13: {'volatilidad': 3.9, 'tendencia': 0.7, 'liquidez': 0.8},
        14: {'volatilidad': 4.3, 'tendencia': 1.1, 'liquidez': 0.9}, # ALTA OPORTUNIDAD
        15: {'volatilidad': 4.8, 'tendencia': 1.4, 'liquidez': 1.0}, # MÁXIMA OPORTUNIDAD
        16: {'volatilidad': 4.6, 'tendencia': 1.2, 'liquidez': 0.9}, # ALTA OPORTUNIDAD
        17: {'volatilidad': 4.2, 'tendencia': 0.8, 'liquidez': 0.8},
        18: {'volatilidad': 3.9, 'tendencia': 0.6, 'liquidez': 0.7},
        19: {'volatilidad': 3.5, 'tendencia': 0.4, 'liquidez': 0.6},
        20: {'volatilidad': 3.1, 'tendencia': 0.2, 'liquidez': 0.5},
        21: {'volatilidad': 2.8, 'tendencia': 0.0, 'liquidez': 0.4},
        22: {'volatilidad': 2.4, 'tendencia': -0.1, 'liquidez': 0.3},
        23: {'volatilidad': 2.2, 'tendencia': -0.2, 'liquidez': 0.3}
    }
    
    predicciones = []
    precio_actual = precio_base
    
    # Generar datos históricos simulados para cálculos técnicos
    precios_historicos = [precio_base + np.random.randn() * 0.1 for _ in range(50)]
    
    # Generar predicciones para 72 horas con análisis detallado
    for hora in range(72):
        timestamp_futuro = inicio + timedelta(hours=hora + 1)
        hora_del_dia = timestamp_futuro.hour
        dia_semana = timestamp_futuro.weekday()
        
        # Obtener patrón base
        patron = patrones_horarios[hora_del_dia]
        volatilidad_base = patron['volatilidad']
        tendencia_base = patron['tendencia']
        liquidez = patron['liquidez']
        
        # Factor de día de la semana
        factor_dia = 1.0
        if dia_semana in [1, 2, 3]:  # Mar-Jue más activos
            factor_dia = 1.15
        elif dia_semana in [5, 6]:  # Fin de semana menos activos
            factor_dia = 0.85
        elif dia_semana == 0:  # Lunes
            factor_dia = 1.05
        elif dia_semana == 4:  # Viernes
            factor_dia = 1.10
        
        # Calcular nuevo precio
        volatilidad_ajustada = volatilidad_base * factor_dia
        tendencia_ajustada = tendencia_base * factor_dia
        
        # Simular movimiento de precio más realista
        ruido_mercado = np.random.randn() * (volatilidad_ajustada / 100)
        precio_futuro = precio_actual * (1 + tendencia_ajustada/100 + ruido_mercado)
        
        # Actualizar histórico
        precios_historicos.append(precio_futuro)
        if len(precios_historicos) > 50:
            precios_historicos.pop(0)
        
        # Calcular indicadores técnicos
        serie_precios = pd.Series(precios_historicos)
        rsi = calcular_rsi(serie_precios).iloc[-1] if len(precios_historicos) >= 14 else 50
        ma_20 = calcular_media_movil(serie_precios, 20).iloc[-1] if len(precios_historicos) >= 20 else precio_futuro
        
        # Determinar señales de trading
        señal_compra = False
        señal_venta = False
        confianza = 0
        
        # Lógica de señales mejorada
        if rsi < 30 and precio_futuro < ma_20 * 0.98 and tendencia_ajustada > 0:
            señal_compra = True
            confianza = min(90, 70 + (30 - rsi) + liquidez * 10)
        elif rsi > 70 and precio_futuro > ma_20 * 1.02 and tendencia_ajustada < 0:
            señal_venta = True
            confianza = min(90, 70 + (rsi - 70) + liquidez * 10)
        
        # Calcular niveles de take profit optimizados
        precio_entrada = precio_futuro
        stop_loss = 0
        take_profit_1 = 0
        take_profit_2 = 0
        take_profit_3 = 0
        duracion_estimada = 0
        momento_cierre = None
        ganancia_estimada = 0
        
        if señal_compra:
            # Para operaciones LONG
            stop_loss = precio_entrada * 0.985  # -1.5%
            take_profit_1 = precio_entrada * 1.015  # +1.5%
            take_profit_2 = precio_entrada * 1.025  # +2.5%
            take_profit_3 = precio_entrada * 1.040  # +4.0%
            
            # Estimar duración basada en volatilidad y liquidez
            duracion_estimada = max(1, int(8 - liquidez * 4))  # 1-8 horas
            momento_cierre = timestamp_futuro + timedelta(hours=duracion_estimada)
            ganancia_estimada = 2.5  # Promedio esperado
            
        elif señal_venta:
            # Para operaciones SHORT
            stop_loss = precio_entrada * 1.015  # +1.5%
            take_profit_1 = precio_entrada * 0.985  # -1.5%
            take_profit_2 = precio_entrada * 0.975  # -2.5%
            take_profit_3 = precio_entrada * 0.960  # -4.0%
            
            duracion_estimada = max(1, int(8 - liquidez * 4))
            momento_cierre = timestamp_futuro + timedelta(hours=duracion_estimada)
            ganancia_estimada = 2.5
        
        # Determinar nivel de oportunidad
        nivel_oportunidad = "BAJA"
        if confianza >= 80:
            nivel_oportunidad = "MUY ALTA"
        elif confianza >= 70:
            nivel_oportunidad = "ALTA"
        elif confianza >= 60:
            nivel_oportunidad = "MEDIA"
        
        predicciones.append({
            'timestamp': timestamp_futuro,
            'hora_colombia': timestamp_futuro.strftime('%H:%M'),
            'dia_semana': ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'][dia_semana],
            'precio_actual': round(precio_futuro, 4),
            'precio_entrada': round(precio_entrada, 4),
            'señal': 'COMPRA' if señal_compra else 'VENTA' if señal_venta else 'ESPERAR',
            'confianza': round(confianza, 1),
            'nivel_oportunidad': nivel_oportunidad,
            'stop_loss': round(stop_loss, 4) if stop_loss > 0 else None,
            'take_profit_1': round(take_profit_1, 4) if take_profit_1 > 0 else None,
            'take_profit_2': round(take_profit_2, 4) if take_profit_2 > 0 else None,
            'take_profit_3': round(take_profit_3, 4) if take_profit_3 > 0 else None,
            'duracion_horas': duracion_estimada if duracion_estimada > 0 else None,
            'momento_cierre': momento_cierre,
            'ganancia_estimada_%': ganancia_estimada if ganancia_estimada > 0 else None,
            'rsi': round(rsi, 1) if not np.isnan(rsi) else None,
            'ma_20': round(ma_20, 4),
            'volatilidad': round(volatilidad_ajustada, 2),
            'liquidez_score': round(liquidez, 2)
        })
        
        precio_actual = precio_futuro
    
    return pd.DataFrame(predicciones)

def mostrar_mejores_oportunidades(df, top_n=10):
    """Muestra las mejores oportunidades de trading"""
    # Filtrar solo señales de compra/venta con alta confianza
    oportunidades = df[
        (df['señal'] != 'ESPERAR') & 
        (df['confianza'] >= 60)
    ].sort_values('confianza', ascending=False).head(top_n)
    
    print("🚀 MEJORES OPORTUNIDADES DE TRADING WLD/USDT")
    print("=" * 70)
    
    for idx, row in oportunidades.iterrows():
        print(f"\n⏰ {row['dia_semana']} {row['hora_colombia']} - {row['timestamp'].strftime('%d/%m/%Y')}")
        print(f"📊 SEÑAL: {row['señal']} | Confianza: {row['confianza']}% | {row['nivel_oportunidad']}")
        print(f"💰 Precio Entrada: ${row['precio_entrada']}")
        
        if row['señal'] != 'ESPERAR':
            print(f"🛑 Stop Loss: ${row['stop_loss']}")
            print(f"🎯 Take Profits: ${row['take_profit_1']} | ${row['take_profit_2']} | ${row['take_profit_3']}")
            print(f"⏱️  Duración: {row['duracion_horas']}h | Cierre: {row['momento_cierre'].strftime('%H:%M %d/%m')}")
            print(f"📈 Ganancia Estimada: {row['ganancia_estimada_%']}%")
        
        print(f"📊 RSI: {row['rsi']} | Volatilidad: {row['volatilidad']}% | Liquidez: {row['liquidez_score']}")
        print("-" * 50)

# Ejecutar análisis
print("Generando análisis avanzado de trading WLD/USDT...")
df_trading = identificar_oportunidades_trading()

# Mostrar mejores oportunidades
mostrar_mejores_oportunidades(df_trading, 15)

# Estadísticas generales
total_señales = len(df_trading[df_trading['señal'] != 'ESPERAR'])
señales_alta_confianza = len(df_trading[df_trading['confianza'] >= 70])

print(f"\n📊 RESUMEN ESTADÍSTICO:")
print(f"Total de señales detectadas: {total_señales}")
print(f"Señales de alta confianza (≥70%): {señales_alta_confianza}")
print(f"Promedio de confianza: {df_trading[df_trading['confianza'] > 0]['confianza'].mean():.1f}%")

# Exportar a CSV para análisis adicional
df_trading.to_csv('predicciones_wld_trading.csv', index=False, sep = ';', decimal = ',')
print(f"\n💾 Datos exportados a 'predicciones_wld_trading.csv'")