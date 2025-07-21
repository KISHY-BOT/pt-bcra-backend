import requests
import pandas as pd
import numpy as np
import certifi
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# === BCRA API WRAPPER ===
class BCRADataFetcher:
    BASE_URL = "https://api.bcra.gob.ar"

    def get_exchange_rate(self, currency="USD", days=30):
        url = f"{self.BASE_URL}/estadisticascambiarias/v1.0/Cotizaciones/{currency}?limit={days}"
        print(f"Fetching exchange rate from: {url}")
        response = requests.get(url, verify=certifi.where())
        return response.json()['results']

    def get_monetary_data(self, variable_id):
        url = f"{self.BASE_URL}/estadisticas/v3.0/monetarias/{variable_id}"
        response = requests.get(url, verify=certifi.where())
        return response.json()['results']

    def get_all_monetary_variables(self):
        url = f"{self.BASE_URL}/estadisticas/v3.0/monetarias"
        response = requests.get(url, verify=certifi.where())
        return response.json()['results']

    def get_debtors_data(self, cuit):
        url = f"{self.BASE_URL}/CentralDeDeudores/v1.0/Deudas/{cuit}"
        response = requests.get(url, verify=certifi.where())
        return response.json().get('results', {})

# === SIMPLIFIED ARIMA PREDICTOR ===
class ExchangeRatePredictor:
    def train(self, historical_data):
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
        if data['reserves'] < 35000:
            alerts.append("Reservas bajas")
        if (data['blue_rate'] - data['official_rate']) / data['official_rate'] > 0.15:
            alerts.append("Brecha cambiaria crítica")
        for d in data['debtors']:
            if d.get("situacion", 1) >= 4:
                alerts.append(f"Empresa en situación {d['situacion']}")
        return alerts

# === OPTIMIZADOR DE PORTAFOLIO ===
class PortfolioOptimizer:
    def optimize(self, risk_level='medium'):
        presets = {
            'low':    {'dolar_mep': 30, 'bonds_cer': 50, 'leliq': 20, 'stocks': 0},
            'medium': {'dolar_mep': 40, 'bonds_cer': 30, 'leliq': 20, 'stocks': 10},
            'high':   {'dolar_mep': 30, 'bonds_cer': 20, 'leliq': 10, 'stocks': 40}
        }
        return presets.get(risk_level, presets['medium'])

# === INSTANCIAS ===
fetcher = BCRADataFetcher()
predictor = ExchangeRatePredictor()
alert_system = EconomicAlertSystem()
optimizer = PortfolioOptimizer()

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
        print("Histórico de cotizaciones:", history[-3:])
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
        return jsonify(fetcher.get_all_monetary_variables())
    except Exception as e:
        print("❌ Error en /variables:", e)
        return jsonify({"error": str(e)}), 500

# === BACKGROUND UPDATE ===
def update_data():
    try:
        current_data['official_rate'] = fetcher.get_exchange_rate(days=1)[0]['tipoCotizacion']
        current_data['reserves'] = fetcher.get_monetary_data(1)[0]['valor']
        current_data['debtors'] = [
            fetcher.get_debtors_data("30500000000"),
            fetcher.get_debtors_data("30600000000")
        ]
        current_data['alerts'] = alert_system.check_alerts(current_data)
        print("✅ Datos actualizados correctamente.")
    except Exception as e:
        print("❌ Error actualizando datos:", e)

scheduler = BackgroundScheduler()
scheduler.add_job(update_data, 'interval', minutes=30)
scheduler.start()

if __name__ == "__main__":
    update_data()
    app.run(host='0.0.0.0', port=5000)
