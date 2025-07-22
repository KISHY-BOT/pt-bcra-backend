import requests
import pandas as pd
import numpy as np
import certifi
import backoff
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# Configuración de seguridad para Render
HEADERS = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
TIMEOUT = 30 # segundos

# === BCRA API WRAPPER MEJORADO ===
class BCRADataFetcher:
BASE_URL = "https://api.bcra.gob.ar"

@backoff.on_exception(backoff.expo,
(requests.exceptions.SSLError, requests.exceptions.ConnectionError),
max_tries=3)
def _make_request(self, url):
response = requests.get(
url,
headers=HEADERS,
verify=certifi.where(),
timeout=TIMEOUT
)
response.raise_for_status()
return response

def get_exchange_rate(self, currency="USD", days=30):
url = f"{self.BASE_URL}/estadisticascambiarias/v1.0/Cotizaciones/{currency}?limit={days}"
print(f"🔍 Fetching exchange rate from: {url}")
response = self._make_request(url)
return response.json()['results']

def get_monetary_data(self, variable_id):
url = f"{self.BASE_URL}/estadisticas/v3.0/monetarias/{variable_id}"
response = self._make_request(url)
return response.json()['results']

def get_all_monetary_variables(self):
url = f"{self.BASE_URL}/estadisticas/v3.0/monetarias"
response = self._make_request(url)
return response.json()['results']

def get_debtors_data(self, cuit):
url = f"{self.BASE_URL}/CentralDeDeudores/v1.0/Deudas/{cuit}"
response = self._make_request(url)
return response.json().get('results', {})

# === SIMPLIFIED ARIMA PREDICTOR ===
class ExchangeRatePredictor:
def train(self, historical_data):
if not historical_data:
raise ValueError("No historical data provided")
df = pd.DataFrame(historical_data)
df['fecha'] = pd.to_datetime(df['fecha'])
df.set_index('fecha', inplace=True)
self.last_value = df['tipoCotizacion'].iloc[-1]
print(f"Último valor cotización entrenado: {self.last_value}")

def predict(self, days=7):
return [round(self.last_value * (1.005 + 0.01*np.random.randn())**i, 2) for i in range(1, days+1)]

# === ALERTAS ===
class EconomicAlertSystem:
def check_alerts(self, data):
alerts = []
# Verificar reservas
if data['reserves'] < 35000:
alerts.append("Reservas bajas")
# Verificar brecha cambiaria (si tenemos blue_rate)
if 'blue_rate' in data and 'official_rate' in data:
if data['blue_rate'] > 0 and data['official_rate'] > 0: # Evitar división por cero
gap = (data['blue_rate'] - data['official_rate']) / data['official_rate']
if gap > 0.15:
alerts.append("Brecha cambiaria crítica")
# Verificar deudores
for d in data['debtors']:
# Solo si 'd' es un diccionario y tiene la clave 'situacion'
if isinstance(d, dict) and d.get("situacion", 1) >= 4:
alerts.append(f"Empresa en situación {d['situacion']}")
return alerts

# === OPTIMIZADOR DE PORTAFOLIO ===
class PortfolioOptimizer:
def optimize(self, risk_level='medium'):
presets = {
'low': {'dolar_mep': 30, 'bonds_cer': 50, 'leliq': 20, 'stocks': 0},
'medium': {'dolar_mep': 40, 'bonds_cer': 30, 'leliq': 20, 'stocks': 10},
'high': {'dolar_mep': 30, 'bonds_cer': 20, 'leliq': 10, 'stocks': 40}
}
return presets.get(risk_level, presets['medium'])

# === INSTANCIAS ===
fetcher = BCRADataFetcher()
predictor = ExchangeRatePredictor()
alert_system = EconomicAlertSystem()
optimizer = PortfolioOptimizer()

# Datos iniciales (se actualizarán en el primer job)
current_data = {
'official_rate': 1280,
'blue_rate': 1300,
'reserves': 39000,
'debtors': [],
'alerts': []
}

# === FLASK ROUTES ===
@app.route("/api/predict/dollar/<int:days>")
def predict_dollar(days):
try:
history = fetcher.get_exchange_rate(days=90)
print(f"Histórico de cotizaciones recibido: {len(history)} registros")
predictor.train(history)
predictions = predictor.predict(days)
return jsonify(predictions)
except Exception as e:
print("❌ Error en /predict/dollar:", e)
return jsonify({"error": str(e)}), 500

@app.route("/api/optimize/<risk_level>")
def optimize_portfolio(risk_level):
return jsonify(optimizer.optimize(risk_level))

@app.route("/api/alerts")
def get_alerts():
return jsonify(current_data['alerts'])

@app.route("/api/variables")
def get_variables():
try:
variables = fetcher.get_all_monetary_variables()
return jsonify(variables)
except Exception as e:
print("❌ Error en /variables:", e)
return jsonify({"error": str(e)}), 500

# === BACKGROUND UPDATE ===
def update_data():
try:
print("Iniciando actualización de datos...")
# Obtener tipo de cambio oficial (último día)
exchange_data = fetcher.get_exchange_rate(days=1)
if exchange_data:
current_data['official_rate'] = exchange_data[0]['tipoCotizacion']

# Obtener reservas (variable ID 1)
reserves_data = fetcher.get_monetary_data(1)
if reserves_data:
current_data['reserves'] = reserves_data[0]['valor']

# Obtener datos de deudores (CUITs de ejemplo)
# NOTA: Reemplaza con CUITs reales de empresas relevantes
current_data['debtors'] = []
for cuit in ["30500000000", "30600000000"]: # Ejemplos
debtor_data = fetcher.get_debtors_data(cuit)
if debtor_data:
current_data['debtors'].append(debtor_data)

# Generar alertas
current_data['alerts'] = alert_system.check_alerts(current_data)
print(f"✅ Datos actualizados. Alertas: {current_data['alerts']}")
except Exception as e:
print("❌ Error actualizando datos:", e)

# Iniciar scheduler
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(update_data, 'interval', minutes=30)
scheduler.start()

# Ejecutar una actualización al inicio
update_data()

if __name__ == "__main__":
app.run(host='0.0.0.0', port=5000)
