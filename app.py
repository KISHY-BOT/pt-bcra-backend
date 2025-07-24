# Aplicar parche de gevent PRIMERO, antes de cualquier otra importación
from gevent import monkey
monkey.patch_all(ssl=False)  # Deshabilitar monkey patching para SSL

import os
import sys
import requests
import pandas as pd
import numpy as np
import certifi
import backoff
import logging
import urllib3
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

# Deshabilitar advertencias de SSL no verificadas
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Aumentar límite de recursión
sys.setrecursionlimit(10000)

# Configuración robusta de certificados
ca_bundle = os.getenv('REQUESTS_CA_BUNDLE', certifi.where())
os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle
os.environ['SSL_CERT_FILE'] = ca_bundle

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                          max_tries=3,
                          jitter=backoff.full_jitter,
                          max_time=30)
    def _make_request(self, url):
        try:
            logger.info(f"Request to: {url}")
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT,
                verify=ca_bundle  # Usar el bundle configurado
            )
            response.raise_for_status()
            return response
        except requests.exceptions.SSLError as e:
            logger.warning(f"SSL Error: {e}. Retrying without verification...")
            # Usar sesión temporal sin verificación
            return requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_exchange_rate(self, currency="USD", days=30):
        url = f"{self.BASE_URL}/estadisticascambiarias/v1.0/Cotizaciones/{currency}?limit={days}"
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
            raise ValueError("No historical data provided")

        df = pd.DataFrame(historical_data)
        if 'fecha' not in df.columns or 'tipoCotizacion' not in df.columns:
            raise ValueError("Incomplete historical data")

        df['fecha'] = pd.to_datetime(df['fecha'])
        df.set_index('fecha', inplace=True)
        self.last_value = df['tipoCotizacion'].iloc[-1]
        logger.info(f"Trained last exchange rate: {self.last_value}")
        return True

    def predict(self, days=7):
        if not hasattr(self, 'last_value'):
            raise ValueError("Model not trained")

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
            "/env": "[DEBUG] Environment variables"
        }
    })

@app.route("/api/predict/dollar/<int:days>")
def predict_dollar(days):
    if days < 1 or days > 30:
        return jsonify({"error": "Invalid range. Use 1-30 days"}), 400

    try:
        history = fetcher.get_exchange_rate(days=90)
        if not history:
            return jsonify({"error": "Could not get historical data"}), 500

        logger.info(f"Received exchange history: {len(history)} records")
        predictor.train(history)
        predictions = predictor.predict(days)
        return jsonify({
            "currency": "USD",
            "predictions": predictions,
            "last_updated": pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in /predict/dollar: {str(e)}")
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
        logger.error(f"Error in /variables: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/env")
def env_vars():
    return jsonify({
        "REQUESTS_CA_BUNDLE": os.getenv("REQUESTS_CA_BUNDLE"),
        "PYTHON_VERSION": os.getenv("PYTHON_VERSION"),
        "SSL_CERT_FILE": os.getenv("SSL_CERT_FILE")
    })

# === ACTUALIZACIÓN EN SEGUNDO PLANO ===
def update_data():
    try:
        logger.info("\n" + "="*50)
        logger.info("Starting data update...")

        exchange_data = fetcher.get_exchange_rate(days=1)
        if exchange_data:
            current_data['official_rate'] = exchange_data[0].get('tipoCotizacion', 1280)
            logger.info(f"✅ Official dollar: ${current_data['official_rate']}")
        else:
            logger.warning("⚠️ Could not get official dollar")

        reserves_data = fetcher.get_monetary_data(1)
        if reserves_data:
            current_data['reserves'] = reserves_data[0].get('valor', 39000)
            logger.info(f"✅ Reserves: USD {current_data['reserves']}M")
        else:
            logger.warning("⚠️ Could not get reserves")

        current_data['debtors'] = []
        company_cuits = ["30500000000", "30600000000"]

        for cuit in company_cuits:
            try:
                debtor_data = fetcher.get_debtors_data(cuit)
                if debtor_data:
                    current_data['debtors'].append(debtor_data)
                    logger.info(f"✅ Debtor data: {cuit}")
            except Exception as e:
                logger.warning(f"⚠️ Error getting debtor {cuit}: {str(e)}")

        current_data['alerts'] = alert_system.check_alerts(current_data)

        if current_data['alerts']:
            logger.info(f"🚨 Active alerts: {len(current_data['alerts'])}")
            for alert in current_data['alerts']:
                logger.info(f"  - {alert}")
        else:
            logger.info("✅ No active alerts")

        logger.info("="*50 + "\n")

    except Exception as e:
        logger.error(f"❌ Critical update error: {str(e)}")

# Configurar scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_data, 'interval', minutes=30)

# Iniciar scheduler inmediatamente
if not scheduler.running:
    scheduler.start()
    update_data()

# Ejecutar aplicación
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("🚀 Starting BCRA Predictor App")
    logger.info(f"Using CA bundle: {ca_bundle}")
    logger.info("="*50)

    app.run(host='0.0.0.0', port=5000)
