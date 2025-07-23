import os
import requests
import pandas as pd
import numpy as np
import certifi
import backoff
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

# Configuración robusta de certificados
ca_bundle = os.getenv('REQUESTS_CA_BUNDLE') or certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle

app = Flask(__name__)

# Configuración de seguridad
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json'
}
TIMEOUT = 30

# === BCRA API WRAPPER ===
class BCRADataFetcher:
    BASE_URL = "https://api.bcra.gob.ar"

    @backoff.on_exception(backoff.expo,
                          (requests.exceptions.RequestException,),
                          max_tries=4,
                          jitter=backoff.full_jitter)
    def _make_request(self, url):
        ssl_path = os.getenv('REQUESTS_CA_BUNDLE')
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT,
                verify=ssl_path
            )
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as e:
            print(f"⚠️ Error SSL: {e}. Reintentando sin verificación...")
            return requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)

    def get_exchange_rate(self, currency="USD", days=30):
        url = f"{self.BASE_URL}/estadisticascambiarias/v1.0/Cotizaciones/{currency}?limit={days}"
        print(f"🔍 Fetching exchange rate from: {url}")
        response = self._make_request(url)
        return response.json().get('results', [])

    def get_monetary_data(self, variable_id):
        url = f"{self.BASE_URL}/estadisticas/v3.0/monetarias/{variable_id}"
        response = self._make_request(url)
        return response.json().get('results', [])

    def get_all_monetary_variables(self):
        url = f"{self.BASE_URL}/estadisticas/v3.0/monetarias"
        response = self._make_request(url)
        return response.json().get('results', [])

    def get_debtors_data(self, cuit):
        url = f"{self.BASE_URL}/CentralDeDeudores/v1.0/Deudas/{cuit}"
        response = self._make_request(url)
        return response.json().get('results', {})

# === PREDICTOR ===
class ExchangeRatePredictor:
    def train(self, historical_data):
        if not historical_data:
            raise ValueError("No se proporcionaron datos históricos")

        df = pd.DataFrame(historical_data)
        if 'fecha' not in df.columns or 'tipoCotizacion' not in df.columns:
            raise ValueError("Datos históricos incompletos")

        df['fecha'] = pd.to_datetime(df['fecha'])
        df.set_index('fecha', inplace=True)
        self.last_value = df['tipoCotizacion'].iloc[-1]
        print(f"Último valor de cotización entrenado: {self.last_value}")
        return True

    def predict(self, days=7):
        if not hasattr(self, 'last_value'):
            raise ValueError("Modelo no entrenado")

        return [round(self.last_value * (1.005 + 0.01*np.random.randn())**i, 2)
                for i in range(1, days+1)]

# === ALERTAS ===
class EconomicAlertSystem:
    def check_alerts(self, data):
        alerts = []

        reserves = data.get('reserves', 0)
        if reserves < 35000:
            alerts.append(f"Reservas bajas (${reserves}M)")

        blue_rate = data.get('blue_rate', 0)
        official_rate = data.get('official_rate', 0)

        if blue_rate > 0 and official_rate > 0:
            gap = (blue_rate - official_rate) / official_rate
            if gap > 0.15:
                alerts.append(f"Brecha cambiaria crítica: {gap:.2%}")

        debtors = data.get('debtors', [])
        for debtor in debtors:
            if isinstance(debtor, dict):
                situation = debtor.get('situacion', 0)
                if situation >= 4:
                    name = debtor.get('denominacion', 'Empresa')
                    alerts.append(f"{name} en situación crediticia {situation}")

        return alerts

# === OPTIMIZADOR ===
class PortfolioOptimizer:
    def optimize(self, risk_level='medium'):
        presets = {
            'low': {'dolar_mep': 30, 'bonds_cer': 50, 'leliq': 20, 'stocks': 0},
            'medium': {'dolar_mep': 40, 'bonds_cer': 30, 'leliq': 20, 'stocks': 10},
            'high': {'dolar_mep': 30, 'bonds_cer': 20, 'leliq': 10, 'stocks': 40}
        }
        return presets.get(risk_level.lower(), presets['medium'])

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

# === RUTAS FLASK ===
@app.route("/")
def home():
    return jsonify({
        "status": "active",
        "endpoints": {
            "/api/predict/dollar/<days>": "Predicción dólar (1-30 días)",
            "/api/optimize/<risk_level>": "Portafolio (low/medium/high)",
            "/api/alerts": "Alertas económicas",
            "/api/variables": "Variables BCRA",
            "/env": "[DEBUG] Ver variables de entorno"
        }
    })

@app.route("/api/predict/dollar/<int:days>")
def predict_dollar(days):
    if days < 1 or days > 30:
        return jsonify({"error": "Rango inválido. Usa 1-30 días"}), 400

    try:
        history = fetcher.get_exchange_rate(days=90)
        if not history:
            return jsonify({"error": "No se pudieron obtener datos históricos"}), 500

        print(f"Histórico de cotizaciones recibido: {len(history)} registros")
        predictor.train(history)
        predictions = predictor.predict(days)
        return jsonify({
            "currency": "USD",
            "predictions": predictions,
            "last_updated": pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        print(f"❌ Error en /predict/dollar: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/optimize/<risk_level>")
def optimize_portfolio(risk_level):
    try:
        portfolio = optimizer.optimize(risk_level)
        return jsonify({
            "risk_level": risk_level,
            "portfolio": portfolio
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/alerts")
def get_alerts():
    return jsonify({
        "alerts": current_data['alerts'],
        "count": len(current_data['alerts'])
    })

@app.route("/api/variables")
def get_variables():
    try:
        variables = fetcher.get_all_monetary_variables()
        return jsonify({
            "count": len(variables),
            "variables": variables
        })
    except Exception as e:
        print(f"❌ Error en /variables: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/env")
def env_vars():
    return jsonify({
        "REQUESTS_CA_BUNDLE": os.getenv("REQUESTS_CA_BUNDLE"),
        "PYTHON_VERSION": os.getenv("PYTHON_VERSION")
    })

# === ACTUALIZACIÓN EN SEGUNDO PLANO ===
def update_data():
    try:
        print("\n" + "="*50)
        print("🔁 Iniciando actualización de datos...")

        exchange_data = fetcher.get_exchange_rate(days=1)
        if exchange_data:
            current_data['official_rate'] = exchange_data[0].get('tipoCotizacion', 1280)
            print(f"✅ Dólar oficial: ${current_data['official_rate']}")
        else:
            print("⚠️ No se pudo obtener el dólar oficial")

        reserves_data = fetcher.get_monetary_data(1)
        if reserves_data:
            current_data['reserves'] = reserves_data[0].get('valor', 39000)
            print(f"✅ Reservas: USD {current_data['reserves']}M")
        else:
            print("⚠️ No se pudieron obtener las reservas")

        current_data['debtors'] = []
        company_cuits = ["30500000000", "30600000000"]

        for cuit in company_cuits:
            try:
                debtor_data = fetcher.get_debtors_data(cuit)
                if debtor_data:
                    current_data['debtors'].append(debtor_data)
                    print(f"✅ Datos de deudor: {cuit}")
            except Exception as e:
                print(f"⚠️ Error obteniendo deudor {cuit}: {str(e)}")

        current_data['alerts'] = alert_system.check_alerts(current_data)

        if current_data['alerts']:
            print(f"🚨 Alertas activas: {len(current_data['alerts'])}")
            for alert in current_data['alerts']:
                print(f"  - {alert}")
        else:
            print("✅ Sin alertas activas")

        print("="*50 + "\n")

    except Exception as e:
        print(f"❌ Error crítico en actualización: {str(e)}")

# Configurar scheduler
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(update_data, 'interval', minutes=30)
scheduler.start()

# Ejecutar actualización inicial
print("="*50)
print("🚀 Iniciando aplicación BCRA Predictor")
print("="*50)
update_data()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
