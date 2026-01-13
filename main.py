import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import sys
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN DE USUARIO (EDITAR AQUÍ)
# ==============================================================================
API_KEY = 'TU_API_KEY_DE_BINANCE'
SECRET_KEY = 'TU_SECRET_KEY_DE_BINANCE'

SYMBOL = 'ETH/USDT'       # Moneda a operar
TIMEFRAME = '15m'         # Temporalidad (debe coincidir con el entrenamiento)
LEVERAGE = 5
MAX_HOLD_HOURS = 8             # Apalancamiento (x5)
RISK_PER_TRADE = 1     # Arriesgar 100% del capital total por operación
CONFIDENCE_THRESHOLD = 0.65 # Solo operar si la IA tiene >65% de certeza

# Objetivos de la estrategia (Hit & Run)
TP_PCT = 0.015  # Ganancia 1.5%
SL_PCT = 0.005  # Pérdida 0.5%

# ==============================================================================
# 2. CONEXIÓN AL EXCHANGE
# ==============================================================================
print(f"--- Iniciando Bot IA [Solo Volumen/Tiempo] ---")
try:
    exchange = ccxt.binanceusdm({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    # exchange.set_sandbox_mode(True) # Descomentar para usar Testnet
    print(f"Conectado exitosamente a Binance Futures.")
except Exception as e:
    print(f"Error crítico de conexión: {e}")
    sys.exit()

# Cargar el modelo XGBoost
model = xgb.XGBClassifier()
try:
    # Asegúrate de que este nombre coincida con el que guardaste al entrenar
    model.load_model("crypto.momentum_25tp.json") 
    print("Modelo IA cargado correctamente.")
except Exception as e:
    print(f"ERROR: No se encuentra el modelo entrenado. {e}")
    sys.exit()

# ==============================================================================
# 3. PROCESAMIENTO DE DATOS (FEATURES)
# ==============================================================================
def prepare_features(df):
    """
    Genera EXACTAMENTE las mismas features que usaste en el entrenamiento.
    Solo: Hora, Día, Cambio de Volumen, Volumen Relativo.
    """
    df = df.copy()
    
    # Features de Tiempo
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    
    # Features de Volumen

    
    # Limpieza de NaN (generados por pct_change y rolling)
    df = df.dropna()
    
    # Selección estricta de columnas para la IA
    features = ['DayOfWeek',"Hour",'Volume']
    return df[features]

# ==============================================================================
# 4. GESTIÓN DE RIESGO
# ==============================================================================
def calculate_position_size(entry_price, balance_usdt):
    # Riesgo en Dólares = Balance * 2%
    risk_amount = balance_usdt * RISK_PER_TRADE
    
    # Distancia al Stop Loss = 0.5%
    stop_distance_pct = SL_PCT
    
    # Tamaño de posición (Notional Value) = Riesgo / Distancia SL
    # Ej: $20 / 0.007 = $2857 de posición total
    position_size_usdt = risk_amount / stop_distance_pct
    
    # Límite de seguridad: No exceder el apalancamiento máximo de la cuenta
    max_buying_power = balance_usdt * LEVERAGE
    
    if position_size_usdt > max_buying_power:
        position_size_usdt = max_buying_power
        print(f"AVISO: Tamaño ajustado al máximo poder de compra ({LEVERAGE}x).")
        
    # Calcular cantidad de monedas
    amount_coins = position_size_usdt / entry_price
    return amount_coins

# ==============================================================================
# 5. LÓGICA PRINCIPAL (EJECUCIÓN)
# ==============================================================================
def run_strategy():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Escaneando mercado...")
    
    # 1. Verificar Posiciones Abiertas
    # No queremos abrir una segunda operación si ya tenemos una corriendo
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for p in positions:
            if float(p['contracts']) > 0:
                pnl = p['unrealizedPnl']
                print(f">> Posición actual en curso. PnL: {pnl} USDT. Esperando cierre...")
                return # Salimos de la función, esperamos al próximo ciclo
    except Exception as e:
        print(f"Error leyendo posiciones: {e}")
        return
    
    entry_time_ms = p['timestamp'] 
    current_time_ms = exchange.milliseconds()
    
    elapsed_hours = (current_time_ms - entry_time_ms) / (1000 * 60 * 60)
    pnl = float(p['unrealizedPnl'])
    print(f">> Posición abierta hace {elapsed_hours:.2f} horas. PnL: {pnl} USDT")

    # B. Lógica de "Time Stop" (Matar al Zombi)
    if elapsed_hours >= MAX_HOLD_HOURS:
        print(f"⚠️ ALERTA: La posición ha durado más de {MAX_HOLD_HOURS} horas.")
        print(">>> EJECUTANDO CIERRE FORZADO (TIME STOP) <<<")
                    
        # 1. Cancelar todas las órdenes pendientes (TP y SL antiguos)
        exchange.cancel_all_orders(SYMBOL)
        print("Órdenes TP/SL canceladas.")

    # 2. Obtener Datos (OHLCV)
    try:
        # Descargamos 50 velas (suficiente para el Vol_SMA de 20)
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        current_price = df['close'].iloc[-1]
    except Exception as e:
        print(f"Error descargando datos: {e}")
        return

    # 3. Preparar Datos para la IA
    try:
        features_df = prepare_features(df)
        last_row = features_df.iloc[[-1]] # Última vela cerrada
    except IndexError:
        print("Datos insuficientes para calcular indicadores.")
        return

    # 4. Predicción de IA
    # predict_proba devuelve [prob_bajada, prob_subida]
    probability = model.predict_proba(last_row)[0][1]
    
    print(f"Precio: {current_price} | Probabilidad IA: {probability:.2%} | Threshold: {CONFIDENCE_THRESHOLD}")

    # 5. Ejecución de Órdenes
    if probability >= CONFIDENCE_THRESHOLD:
        print(">>> ¡SEÑAL VÁLIDA DETECTADA! EJECUTANDO COMPRA <<<")
        
        try:
            # A. Configurar Apalancamiento
            exchange.set_leverage(LEVERAGE, SYMBOL)
            
            # B. Calcular Tamaño
            balance = exchange.fetch_balance()['USDT']['free']
            amount = calculate_position_size(current_price, balance)
            
            # Ajustar precisión para Binance
            amount = exchange.amount_to_precision(SYMBOL, amount)
            
            # C. Enviar Orden de Entrada (MARKET)
            order = exchange.create_market_buy_order(SYMBOL, amount)
            real_entry_price = float(order['average'])
            print(f"COMPRA completada a: {real_entry_price}")
            
            # D. Calcular TP y SL precios
            tp_price = real_entry_price * (1 + TP_PCT)
            sl_price = real_entry_price * (1 - SL_PCT)
            
            # Ajustar precisión de precios
            tp_price = float(exchange.price_to_precision(SYMBOL, tp_price))
            sl_price = float(exchange.price_to_precision(SYMBOL, sl_price))
            
            # E. Enviar STOP LOSS (Reduce Only)
            sl_params = {'stopPrice': sl_price, 'reduceOnly': True}
            exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', amount, None, params=sl_params)
            print(f"STOP LOSS colocado en: {sl_price}")
            
            # F. Enviar TAKE PROFIT (Reduce Only)
            tp_params = {'stopPrice': tp_price, 'reduceOnly': True}
            exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', amount, None, params=tp_params)
            print(f"TAKE PROFIT colocado en: {tp_price}")
            
            print("--- Operación configurada correctamente ---")
            
        except Exception as e:
            print(f"¡ERROR CRÍTICO AL OPERAR!: {e}")
    else:
        print("Señal débil. Esperando...")

# ==============================================================================
# 6. BUCLE INFINITO
# ==============================================================================
if __name__ == "__main__":
    while True:
        run_strategy()
        
        # Esperar 15 minutos (900 segundos) para la siguiente vela
        # Recomendación: Poner 60 segundos si quieres que revise constantemente 
        # (aunque la vela no cierre), pero para XGBoost es mejor al cierre.
        print("Durmiendo 1 minuto hasta el próximo chequeo...")
        time.sleep(900)