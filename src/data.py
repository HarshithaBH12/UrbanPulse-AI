from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

def fetch_live_city_data(city: str, lat: float, lon: float) -> dict:
    """Fetch current weather and European AQI from Open-Meteo; use safe fallback values if offline."""
    result = {
        "temperature": 27.0, "humidity": 60.0, "wind_speed": 8.0,
        "rainfall": 0.0, "aqi": 45.0, "api_error": False
    }
    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "Asia/Kolkata"
        }
        response = requests.get(weather_url, params=params, timeout=8)
        response.raise_for_status()
        current = response.json().get("current", {})
        result["temperature"] = float(current.get("temperature_2m", result["temperature"]))
        result["humidity"] = float(current.get("relative_humidity_2m", result["humidity"]))
        result["rainfall"] = float(current.get("precipitation", result["rainfall"]))
        result["wind_speed"] = float(current.get("wind_speed_10m", result["wind_speed"]))

        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_params = {
            "latitude": lat, "longitude": lon, "current": "european_aqi",
            "timezone": "Asia/Kolkata"
        }
        air_response = requests.get(air_url, params=air_params, timeout=8)
        air_response.raise_for_status()
        air_current = air_response.json().get("current", {})
        aqi = air_current.get("european_aqi")
        result["aqi"] = float(aqi) if aqi is not None else None
    except (requests.RequestException, ValueError, TypeError, KeyError):
        result["api_error"] = True

    # Clearly synthetic operations signal, derived only for demonstration.
    hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour
    rush = 18 if hour in range(7, 11) or hour in range(16, 21) else 0
    rain_penalty = min(result["rainfall"] * 1.5, 15)
    result["energy_index"] = float(np.clip(48 + max(result["temperature"] - 24, 0) * 1.2 + rush * .25, 15, 95))
    result["traffic_seed"] = float(np.clip(35 + rush + rain_penalty, 10, 95))
    return result

def build_historical_dataset(city: str, live: dict, rows: int = 720) -> pd.DataFrame:
    """Create reproducible synthetic hourly observations, with a small live-signal influence."""
    seed = sum(ord(ch) for ch in city) + 2026
    rng = np.random.default_rng(seed)
    end = datetime.now(ZoneInfo("Asia/Kolkata")).replace(minute=0, second=0, microsecond=0)
    timestamps = [end - timedelta(hours=rows - 1 - i) for i in range(rows)]
    hours = np.array([t.hour for t in timestamps])
    weekdays = np.array([t.weekday() for t in timestamps])
    is_weekend = (weekdays >= 5).astype(int)
    temperature = np.clip(26 + 5*np.sin((hours-7)*np.pi/12) + rng.normal(0, 2.2, rows), 15, 43)
    rainfall = np.clip(rng.gamma(1.0, 1.2, rows) - .8, 0, 20)
    aqi_base = live.get("aqi") if live.get("aqi") is not None else 50
    aqi = np.clip(aqi_base + rng.normal(0, 16, rows), 5, 200)
    rush = np.where((hours >= 7) & (hours <= 10), 24, 0) + np.where((hours >= 16) & (hours <= 20), 30, 0)
    weekend_reduction = is_weekend * 12
    event_factor = rng.choice([0, 0, 0, 8, 15, 20], rows)
    traffic = 28 + rush - weekend_reduction + rainfall*0.7 + event_factor + (aqi/200)*5 + rng.normal(0, 7, rows)
    traffic = np.clip(traffic, 5, 100)
    energy = np.clip(40 + np.maximum(temperature-23, 0)*1.6 + np.where((hours>=9)&(hours<=18), 12, 0) + rng.normal(0,5,rows), 10, 100)
    return pd.DataFrame({
        "timestamp": timestamps, "city": city, "hour": hours, "is_weekend": is_weekend,
        "temperature": temperature.round(2), "rainfall": rainfall.round(2), "aqi": aqi.round(1),
        "event_factor": event_factor, "traffic_index": traffic.round(2), "energy_index": energy.round(2),
        "humidity": np.clip(live.get("humidity", 60) + rng.normal(0, 8, rows), 20, 100).round(1)
    })
