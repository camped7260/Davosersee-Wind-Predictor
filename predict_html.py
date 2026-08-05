#!/usr/bin/env python3
"""
predict.py

Standalone prediction & HTML dashboard generator for Wingfoil forecasting (Davos).
Runs predictions for today and tomorrow by loading 
weights, features, scalers, and offsets dynamically from `weights.json`.
"""

import argparse
import io
import json
import math
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Import foehn pressure gradient calculation function identically to main script
from foehn_gradient import get_combined_data_foehn_gradient

# =====================================================================
# CONFIGURATION & WEIGHTS LOADING
# =====================================================================

CONFIG = {
    "settings": {
        "timezone": "Europe/Zurich",
        "wind_threshold_knots": 10.0,
        "foil_confirmed_threshold_knots": 10.0,
        "operational_window_utc": [11, 17],
        "init_valley_angle": 10.0
    },
    "locations": {
        "davos": {"lat": 46.8041, "lon": 9.8372, "station_id": "06784"}
    }
}

# Fallback weights in case weights.json is missing
DEFAULT_WEIGHTS = {
    "version": "v2.5-json",
    "updated_at": "2026-08-05 00:00:00 UTC",
    "global_fallback_bias": 0.45,
    "global_fx1_fallback_bias": 1.20,
    "category_mean_offsets": {
        "Sunny": 0.52,
        "PartlyCloudy": 0.38,
        "Foehn + Sunny": -1.15,
        "Foehn + PartlyCloudy": -0.85,
        "Cloudy": 0.12
    },
    "category_fx1_mean_offsets": {
        "Sunny": 1.10,
        "PartlyCloudy": 0.95,
        "Foehn + Sunny": -0.40,
        "Foehn + PartlyCloudy": -0.20,
        "Cloudy": 0.50
    },
    "category_angles": {
        "DEFAULT": 10.0,
        "Foehn + Sunny": 12.0,
        "Foehn + PartlyCloudy": 12.0
    },
    "bayesian_models": {
        "Sunny": {
            "features": ["om_syn_ff_kt", "om_syn_dd_along_valley", "om_syn_dd_cross_valley"],
            "scaler_mean": [12.4, 0.15, -0.05],
            "scaler_scale": [4.2, 0.65, 0.70],
            "coef": [0.85, -0.42, 0.18],
            "intercept": 0.48
        },
        "PartlyCloudy": {
            "features": ["mosmix_ff_kt", "mosmix_rad_kj", "om_syn_dd_cross_valley"],
            "scaler_mean": [8.5, 1800.0, 0.02],
            "scaler_scale": [3.1, 600.0, 0.68],
            "coef": [0.62, -0.25, 0.12],
            "intercept": 0.35
        },
        "Foehn + PartlyCloudy": {
            "features": ["mosmix_dp_foehn", "mosmix_cloud_pct", "om_prec_prob"],
            "scaler_mean": [5.2, 45.0, 15.0],
            "scaler_scale": [2.1, 15.0, 10.0],
            "coef": [-1.45, 0.35, -0.15],
            "intercept": -0.92
        }
    },
    "bayesian_fx1_models": {
        "Sunny": {
            "features": ["mosmix_fx1_kt", "om_syn_ff_kt", "om_syn_dd_along_valley"],
            "scaler_mean": [18.2, 12.4, 0.15],
            "scaler_scale": [5.1, 4.2, 0.65],
            "coef": [0.92, 0.35, -0.20],
            "intercept": 1.05
        },
        "Foehn + PartlyCloudy": {
            "features": ["mosmix_fx1_kt", "mosmix_dp_foehn", "mosmix_cloud_pct"],
            "scaler_mean": [22.0, 5.2, 45.0],
            "scaler_scale": [6.0, 2.1, 15.0],
            "coef": [0.78, -0.95, 0.18],
            "intercept": -0.15
        }
    }
}

def load_exported_weights(json_path="model_weights.json"):
    """Loads feature parameters and weights from JSON file or uses defaults."""
    weights_file = Path(json_path)
    if weights_file.exists():
        try:
            with open(weights_file, "r", encoding="utf-8") as f:
                print(f"⚙️ Loaded model parameters and weights from {weights_file.resolve()}")
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading {json_path}: {e}. Falling back to default weights.")
    else:
        print(f"ℹ️ {json_path} not found. Utilizing default fallback parameters.")
    return DEFAULT_WEIGHTS

EXPORTED_WEIGHTS = load_exported_weights("model_weights.json")

# Extract version string using version and updated_at timestamp from the weights JSON
VERSION = f"{EXPORTED_WEIGHTS.get('version', 'v2.5-json')} ({EXPORTED_WEIGHTS.get('updated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))})"

REGIME_COLORS = {
    "Sunny": "#fff9db",
    "PartlyCloudy": "#f1f3f5",
    "Cloudy": "#e9ecef",
    "Foehn": "#ffe3e3",
    "Foehn + Sunny": "#ffe8cc",
    "Foehn + PartlyCloudy": "#ffdeeb",
}
DEFAULT_REGIME_COLOR = "#ffffff"

# =====================================================================
# HELPER CONVERSIONS & UTILITIES
# =====================================================================

def convert_ms_to_knots(ms):
    return ms * 1.94384 if ms is not None else None

def convert_kmh_to_knots(kmh):
    return kmh / 1.852 if kmh is not None else None

def kelvin_to_celsius(k):
    return k - 273.15 if k is not None else None

def degrees_to_cardinal(deg):
    """Converts wind direction degrees to a 16-point compass direction string."""
    if deg is None or pd.isna(deg):
        return "-"
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((deg + 11.25) / 22.5) % 16
    return f"{dirs[idx]} ({deg:.0f}°)"

def clean_namespaces(root):
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        for key in list(elem.attrib.keys()):
            if '}' in key:
                new_key = key.split('}', 1)[1]
                elem.attrib[new_key] = elem.attrib.pop(key)
    return root

# =====================================================================
# STANDALONE PREDICTION ENGINE
# =====================================================================

class StandaloneWindPredictor:
    def __init__(self, weights=EXPORTED_WEIGHTS):
        self.w = weights
        self.foil_threshold = CONFIG["settings"]["foil_confirmed_threshold_knots"]

    @staticmethod
    def classify_precip_type(row):
        if not row.get("is_precip", False):
            return "No_Precip"
        
        w_code = row.get("om_w_codes")
        if pd.isna(w_code):
            return "Precip_Unknown"
            
        w_code = int(w_code)
        if (50 <= w_code <= 69) or (80 <= w_code <= 82):
            return "Rain"
        elif (70 <= w_code <= 79) or (85 <= w_code <= 86):
            return "Snow"
        elif 90 <= w_code <= 99:
            return "Thunderstorm"
        else:
            return "Other_Precip"

    def _prepare_features(self, df):
        df = df.copy()
        
        # Calculate foehn pressure gradient using target date string(s)
        if "mosmix_dp_foehn" not in df.columns or df["mosmix_dp_foehn"].isna().all():
            try:
                unique_dates = df.index.strftime("%Y-%m-%d").unique()
                foehn_records = []
                
                for target_date in unique_dates:
                    records = get_combined_data_foehn_gradient(target_date)
                    if records:
                        foehn_records.extend(records)

                if foehn_records:
                    df_foehn = (
                        pd.DataFrame(foehn_records)
                        .drop_duplicates(subset=["datetime"], keep="first")
                        .set_index("datetime")
                    )
                    df["mosmix_dp_foehn"] = df.index.map(df_foehn["dp_foehn"])

                if df["mosmix_dp_foehn"].isna().any():
                    df["mosmix_dp_foehn"] = df["mosmix_dp_foehn"].fillna(
                        df["mosmix_u_kt"].apply(lambda u: u * 0.4 if pd.notna(u) else 0.0)
                    )

            except Exception as e:
                print(f"⚠️ Warning: Could not calculate foehn gradient ({e}). Falling back to proxy.")
                df["mosmix_dp_foehn"] = df["mosmix_u_kt"].apply(lambda u: u * 0.4 if pd.notna(u) else 0.0)

        df["is_nordfoehn"] = df["mosmix_dp_foehn"] > 3.5
        df["is_sudfoehn"] = df["mosmix_dp_foehn"] < -3.5
        df["is_sunny"] = df["mosmix_cloud_pct"] < 33.0
        df["is_partly_cloudy"] = (df["mosmix_cloud_pct"] >= 33.0) & (df["mosmix_cloud_pct"] <= 66.0)
        df["is_cloudy"] = df["mosmix_cloud_pct"] > 66.0
        df["is_precip"] = df["om_prec_prob"] > 50.0

        df["precip_type"] = df.apply(self.classify_precip_type, axis=1)

        def classify(row):
            tags = []

            # 1. Foehn status
            if row.get("is_nordfoehn", False):
                tags.append("Foehn")
            elif row.get("is_sudfoehn", False):
                tags.append("Sudfoehn")

            # 2. Cloud Cover status
            if row.get("is_sunny", False):
                tags.append("Sunny")
            elif row.get("is_partly_cloudy", False):
                tags.append("PartlyCloudy")
            elif row.get("is_cloudy", False):
                tags.append("Cloudy")

            # 3. Precipitation status
            if row.get("is_precip", False):
                tags.append(f"Precip({row.get('precip_type', 'No_Precip')})")

            return " + ".join(tags) if tags else "Unclassified"

        df["classification"] = df.apply(classify, axis=1)

        angles_map = self.w.get("category_angles", {})
        default_angle = angles_map.get("DEFAULT", CONFIG["settings"]["init_valley_angle"])

        for idx, row in df.iterrows():
            cat = row["classification"]
            angle = angles_map.get(cat, default_angle)
            m_rad = np.radians(row.get("mosmix_dd_deg", 0.0) - angle)
            s_rad = np.radians(row.get("om_syn_dd_deg", 0.0) - angle)
            
            df.loc[idx, "mosmix_dd_along_valley"] = np.cos(m_rad)
            df.loc[idx, "mosmix_dd_cross_valley"] = np.sin(m_rad)
            df.loc[idx, "om_syn_dd_along_valley"] = np.cos(s_rad)
            df.loc[idx, "om_syn_dd_cross_valley"] = np.sin(s_rad)

        return df

    def apply_rain_multiplicative_factor(self, raw_speed, prec_prob, w_code):
        if raw_speed < 1.2:
            return raw_speed, 1.0
        code_known = not pd.isna(w_code)
        is_rain = code_known and ((50 <= int(w_code) <= 69) or (80 <= int(w_code) <= 82))
        
        triggers = (prec_prob > 50.0 and is_rain) or (prec_prob > 75.0) if code_known else prec_prob > 50.0
        floor = 0.75 if code_known else 0.875
        
        if triggers:
            factor = max(floor, 1.0 - ((1.0 - floor) * (prec_prob / 100.0)))
            return raw_speed * factor, factor
        return raw_speed, 1.0

    def predict(self, df_input):
        df = self._prepare_features(df_input)
        corr_ff, std_ff = [], []
        corr_fx, std_fx = [], []
        engines = []

        bayesian_ff = self.w.get("bayesian_models", {})
        bayesian_fx = self.w.get("bayesian_fx1_models", {})
        mean_offsets = self.w.get("category_mean_offsets", {})
        fx1_offsets = self.w.get("category_fx1_mean_offsets", {})

        for idx, row in df.iterrows():
            raw_ff = row["mosmix_ff_kt"]
            raw_fx = row.get("mosmix_fx1_kt", raw_ff)
            cat = row["classification"]
            prec_prob = row.get("om_prec_prob", 0.0)
            w_code = row.get("om_w_codes", np.nan)

            # --- Mean Wind Speed Correction (ff) ---
            rain_ff, factor = self.apply_rain_multiplicative_factor(raw_ff, prec_prob, w_code)
            if factor < 1.0:
                final_ff, s_ff, engine = rain_ff, 0.0, f"Rain ({factor:.2f}x)"
            elif cat in bayesian_ff:
                m = bayesian_ff[cat]
                x_raw = np.array([row.get(f, 0.0) for f in m["features"]])
                s_mean = np.array(m["scaler_mean"])
                s_scale = np.array(m["scaler_scale"])
                s_scale[s_scale == 0] = 1.0
                
                x_scaled = (x_raw - s_mean) / s_scale
                bias = float(np.dot(x_scaled, np.array(m["coef"])) + m["intercept"])
                final_ff, s_ff, engine = raw_ff - bias, 1.2, "Bayesian Ridge"
            elif cat in mean_offsets:
                final_ff, s_ff, engine = raw_ff - mean_offsets[cat], 0.0, "Cat Offset"
            else:
                final_ff, s_ff, engine = raw_ff - self.w.get("global_fallback_bias", 0.45), 0.0, "Global Offset"

            # --- Gust Correction (fx1) ---
            rain_fx, factor_fx = self.apply_rain_multiplicative_factor(raw_fx, prec_prob, w_code)
            if factor_fx < 1.0:
                final_fx, s_fx = rain_fx, 0.0
            elif cat in bayesian_fx:
                m_fx = bayesian_fx[cat]
                x_raw_fx = np.array([row.get(f, 0.0) for f in m_fx["features"]])
                s_mean_fx = np.array(m_fx["scaler_mean"])
                s_scale_fx = np.array(m_fx["scaler_scale"])
                s_scale_fx[s_scale_fx == 0] = 1.0

                x_scaled_fx = (x_raw_fx - s_mean_fx) / s_scale_fx
                bias_fx = float(np.dot(x_scaled_fx, np.array(m_fx["coef"])) + m_fx["intercept"])
                final_fx, s_fx = raw_fx - bias_fx, 1.5
            elif cat in fx1_offsets:
                final_fx, s_fx = raw_fx - fx1_offsets[cat], 0.0
            else:
                final_fx, s_fx = raw_fx - self.w.get("global_fx1_fallback_bias", 1.20), 0.0

            final_ff = float(np.clip(final_ff, 0.0, None))
            final_fx = float(np.clip(final_fx, final_ff, None))

            corr_ff.append(final_ff)
            std_ff.append(s_ff)
            corr_fx.append(final_fx)
            std_fx.append(s_fx)
            engines.append(engine)

        df["mosmix_ff_corrected_kt"] = corr_ff
        df["mosmix_ff_std_kt"] = std_ff
        df["mosmix_fx1_corrected_kt"] = corr_fx
        df["mosmix_fx1_std_kt"] = std_fx
        df["correction_engine"] = engines
        return df

# =====================================================================
# DATA RETRIEVAL (DWD MOSMIX & OPEN-METEO & DSSC)
# =====================================================================

def fetch_mosmix(station_id):
    url = f"https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations/{station_id}/kml/MOSMIX_L_LATEST_{station_id}.kmz"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        res.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(res.content)) as zip_ref:
            kml_name = next(name for name in zip_ref.namelist() if name.endswith('.kml'))
            kml_content = zip_ref.read(kml_name)
        root = clean_namespaces(ET.fromstring(kml_content))

        timesteps = [ts.text for ts in root.findall('.//ForecastTimeSteps/TimeStep')]
        parsed_data = {ts: {"ff": None, "fx1": None, "dd": None, "ttt": None, "rad1h": None, "n": None} for ts in timesteps}
        
        for element in root.findall('.//Forecast'):
            elem_name = element.get('elementName')
            if elem_name in ["FF", "FX1", "DD", "TTT", "Rad1h", "N"]:
                key = elem_name.lower()
                val_elem = element.find('value')
                if val_elem is not None and val_elem.text:
                    for idx, val_str in enumerate(val_elem.text.split()):
                        if idx < len(timesteps):
                            try:
                                v = float(val_str)
                                if v != -999.0: parsed_data[timesteps[idx]][key] = v
                            except ValueError: pass

        records = []
        for ts, vals in parsed_data.items():
            dt_utc = pd.to_datetime(ts)
            dt_local = dt_utc.tz_convert('Europe/Zurich').tz_localize(None) if dt_utc.tz else dt_utc
            u_ms = - vals['ff'] * np.sin(np.radians(vals['dd'])) if vals['ff'] and vals['dd'] else None
            v_ms = - vals['ff'] * np.cos(np.radians(vals['dd'])) if vals['ff'] and vals['dd'] else None
            records.append({
                'datetime': dt_local,
                'mosmix_ff_kt': convert_ms_to_knots(vals['ff']),
                'mosmix_fx1_kt': convert_ms_to_knots(vals['fx1']),
                'mosmix_dd_deg': vals['dd'],
                'mosmix_tt_c': kelvin_to_celsius(vals['ttt']),
                'mosmix_rad_kj': vals['rad1h'] / 1000.0 if vals['rad1h'] else None,
                'mosmix_cloud_pct': vals['n'],
                'mosmix_u_kt': convert_ms_to_knots(u_ms),
                'mosmix_v_kt': convert_ms_to_knots(v_ms),
            })
        df = pd.DataFrame(records).set_index('datetime').sort_index()
        return df
    except Exception as e:
        print(f"⚠️ Error fetching MOSMIX: {e}")
        return None

def fetch_openmeteo(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wind_speed_700hPa,wind_direction_700hPa,weather_code,precipitation_probability",
        "wind_speed_unit": "ms", "timezone": "Europe/Zurich",
        "start_date": start_date, "end_date": end_date
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            h = res.json().get("hourly", {})
            return pd.DataFrame({
                "om_syn_ff_kt": [convert_ms_to_knots(s) for s in h.get("wind_speed_700hPa", [])],
                "om_syn_dd_deg": h.get("wind_direction_700hPa", []),
                "om_prec_prob": h.get("precipitation_probability", []),
                "om_w_codes": h.get("weather_code", [])
            }, index=pd.to_datetime(h.get("time")).tz_localize(None))
    except Exception as e:
        print(f"⚠️ Error fetching Open-Meteo: {e}")
    return None

# =====================================================================
# FETCH & PROCESS DSSC DATA
# =====================================================================
def fetch_dssc_data(endpoint_name):
    url = f"https://www.dssc.ch/cumulusmx/{endpoint_name}.json"
    headers = {'User-Agent': 'Mozilla/5.0 (Ubuntu; Linux x86_64) WingfoilPredictor/1.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"[⚠️ Warning] Station DSSC ({endpoint_name}) indisponible : {e}")
    return None

def process_dssc_hourly(wind_data, wind_dir_data, temp_data, target_date):
    hourly_raw = {f"{h:02d}:00": {"speeds": [], "gusts": [], "dirs": [], "temps": []} for h in range(0, 24)}
    
    if wind_data:
        for timestamp, val_kmh in wind_data.get("wspeed", []):
            dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            time_str = dt.strftime("%H:00")
            if dt.strftime("%Y-%m-%d") == target_date and time_str in hourly_raw:
                hourly_raw[time_str]["speeds"].append(val_kmh)
                
        for timestamp, val_kmh in wind_data.get("wgust", []):
            dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            time_str = dt.strftime("%H:00")
            if dt.strftime("%Y-%m-%d") == target_date and time_str in hourly_raw:
                hourly_raw[time_str]["gusts"].append(val_kmh)

    if wind_dir_data:
        for timestamp, val_deg in wind_dir_data.get("avgbearing", []):
            dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            time_str = dt.strftime("%H:00")
            if dt.strftime("%Y-%m-%d") == target_date and time_str in hourly_raw:
                hourly_raw[time_str]["dirs"].append(val_deg)

    if temp_data:
        temp_key = "temp" if "temp" in temp_data else next(iter(temp_data.keys()), None)
        if temp_key and isinstance(temp_data.get(temp_key), list):
            for timestamp, val_c in temp_data[temp_key]:
                dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                time_str = dt.strftime("%H:00")
                if dt.strftime("%Y-%m-%d") == target_date and time_str in hourly_raw:
                    hourly_raw[time_str]["temps"].append(val_c)
                 
    hourly_obs = {}
    for hour, data in hourly_raw.items():
        if data["speeds"] or data["temps"] or data["dirs"]:
            avg_dir = None
            if data["dirs"]:
                sin_sum = sum(np.sin(np.radians(d)) for d in data["dirs"])
                cos_sum = sum(np.cos(np.radians(d)) for d in data["dirs"])
                R = np.hypot(sin_sum, cos_sum)
                if R > 1e-5:
                    avg_dir = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360)

            hourly_obs[hour] = {
                "speed": convert_kmh_to_knots(sum(data["speeds"]) / len(data["speeds"])) if data["speeds"] else None,
                "gust": convert_kmh_to_knots(max(data["gusts"])) if data["gusts"] else (convert_kmh_to_knots(max(data["speeds"])) if data["speeds"] else None),
                "dir": avg_dir,
                "temp": sum(data["temps"]) / len(data["temps"]) if data["temps"] else None
            }
        else:
            hourly_obs[hour] = None
    return hourly_obs

# =====================================================================
# GRAPH GENERATION
# =====================================================================

def generate_day_graph(date_str, df_day, dssc_obs, output_path):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    hours = df_day.index.hour
    
    # Background Regime Shading
    for h, cat in zip(hours, df_day["classification"]):
        color = REGIME_COLORS.get(cat, DEFAULT_REGIME_COLOR)
        ax.axvspan(h - 0.5, h + 0.5, color=color, alpha=0.45, zorder=0)

    # Wind Speed Signal Lines
    corr_ff = df_day["mosmix_ff_corrected_kt"]
    corr_fx = df_day["mosmix_fx1_corrected_kt"]

    ax.plot(hours, df_day["mosmix_ff_kt"], label="Wind Raw (MOSMIX-L)", color="#95a5a6", linestyle="--", linewidth=1.2)
    ax.plot(hours, corr_ff, label="Wind Corrected", color="#2ecc71", linewidth=2.2)
    ax.plot(hours, corr_fx, label="Gust Corrected", color="#e67e22", linewidth=2.2)

    if dssc_obs:
        obs_h = [
            int(k.split(":")[0]) 
            for k, v in dssc_obs.items() 
            if v is not None and "speed" in v and v["speed"] is not None
        ]
        obs_sp = [
            v["speed"] 
            for k, v in dssc_obs.items() 
            if v is not None and "speed" in v and v["speed"] is not None
        ]
        if obs_h:
            ax.scatter(obs_h, obs_sp, color="#e74c3c", label="Observation", zorder=5, s=30)

    ax.axhline(CONFIG["settings"]["wind_threshold_knots"], color="#e74c3c", linestyle=":", alpha=0.7, label="Threshold (10kt)")
    
    # Limits & Spacing
    ax.set_xlim(10, 19)
    ax.set_ylim(0, 25)
    ax.set_xticks(range(10, 20))
    
    # 1-knot grid ticks on the y-axis
    y_ticks = np.arange(0, 26, 1)
    y_labels = [str(y) if y % 1 == 0 else "" for y in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Local Hour")
    ax.set_ylabel("Wind Speed (knots)")
    ax.set_title(f"Davosersee Forecast — {date_str}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =====================================================================
# HTML DASHBOARD GENERATOR
# =====================================================================

def generate_mobile_html(days_data, output_file="index.html"):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Davosersee Wind Forecast</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 12px; color: #212529; }}
        .header {{ background: #1e293b; color: white; padding: 14px; border-radius: 10px; margin-bottom: 12px; }}
        .header h1 {{ margin: 0; font-size: 1.2rem; }}
        .version {{ font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }}
        .day-card {{ background: white; border-radius: 10px; padding: 12px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow-x: auto; }}
        .day-title {{ font-weight: bold; font-size: 1.1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; color: white; }}
        .bg-go {{ background: #22c55e; }}
        .bg-nogo {{ background: #ef4444; }}
        img {{ width: 100%; border-radius: 6px; margin: 8px 0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.72rem; margin-top: 8px; white-space: nowrap; }}
        th, td {{ padding: 6px 4px; text-align: center; border-bottom: 1px solid #f1f5f9; }}
        th {{ background: #f8fafc; color: #64748b; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏄 Davosersee Wind Forecast by Camille, DWD MOSMIX-L data with corrections</h1>
        <h1>-- experimental, at your own risk, no guarantees! --</h1>
        <div class="version">Generated: {VERSION}</div>
    </div>
"""

    for date_str, data in days_data.items():
        max_ff = data["df"]["mosmix_ff_corrected_kt"].max()
        status_badge = '<span class="badge bg-go">🟢 WINGFOIL</span>' if max_ff >= 10.0 else '<span class="badge bg-nogo">🔴 NO WIND</span>'
        
        html_content += f"""
    <div class="day-card">
        <div class="day-title">
            <span>{date_str}</span>
            {status_badge}
        </div>
        <img src="{data['graph_name']}" alt="Forecast Graph">
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Wind Raw (kt)</th>
                    <th>Wind Corr (kt)</th>
                    <th>Gust Corr (kt)</th>
                    <th>Wind Dir</th>
                    <th>Temp (°C)</th>
                    <th>Cloud (%)</th>
                    <th>Rain Prob (%)</th>
                    <th>Foehn Grad (hPa)</th>
                    <th>Regime</th>
                </tr>
            </thead>
            <tbody>
"""
        for ts, row in data["df"].iterrows():
            if 10 <= ts.hour <= 19:
                raw_ff = f"{row['mosmix_ff_kt']:.1f}" if pd.notna(row.get('mosmix_ff_kt')) else "-"
                corr_ff = f"{row['mosmix_ff_corrected_kt']:.1f}" if pd.notna(row.get('mosmix_ff_corrected_kt')) else "-"
                corr_fx = f"{row['mosmix_fx1_corrected_kt']:.1f}" if pd.notna(row.get('mosmix_fx1_corrected_kt')) else "-"
                wind_dir = degrees_to_cardinal(row.get('mosmix_dd_deg'))
                temp = f"{row['mosmix_tt_c']:.1f}" if pd.notna(row.get('mosmix_tt_c')) else "-"
                cloud = f"{row['mosmix_cloud_pct']:.0f}%" if pd.notna(row.get('mosmix_cloud_pct')) else "-"
                rain = f"{row['om_prec_prob']:.0f}%" if pd.notna(row.get('om_prec_prob')) else "-"
                foehn_grad = f"{row['mosmix_dp_foehn']:.1f}" if pd.notna(row.get('mosmix_dp_foehn')) else "-"
                
                html_content += f"""
                <tr>
                    <td>{ts.strftime('%H:%M')}</td>
                    <td>{raw_ff}</td>
                    <td><b>{corr_ff}</b></td>
                    <td>{corr_fx}</td>
                    <td>{wind_dir}</td>
                    <td>{temp}</td>
                    <td>{cloud}</td>
                    <td>{rain}</td>
                    <td>{foehn_grad}</td>
                    <td>{row.get('classification', '-')}</td>
                </tr>"""

        html_content += """
            </tbody>
        </table>
    </div>"""

    html_content += """
</body>
</html>"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"📱 Mobile HTML dashboard generated: {output_file}")

# =====================================================================
# MAIN EXECUTION ROUTINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone Wingfoil Predictor & Dashboard Generator")
    parser.add_argument("--include-dssc", action="store_true", help="Include realtime DSSC observations")
    parser.add_argument("--weights-file", type=str, default="model_weights.json", help="Path to weights JSON file")
    args = parser.parse_args()

    global EXPORTED_WEIGHTS, VERSION
    EXPORTED_WEIGHTS = load_exported_weights(args.weights_file)
    VERSION = f"{EXPORTED_WEIGHTS.get('version', 'v2.5-json')} ({EXPORTED_WEIGHTS.get('updated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))})"

    print(f"🚀 Running Wingfoil Prediction Engine [{VERSION}]")
    if args.include_dssc:
        print("📡 DSSC Realtime Observations: ENABLED")
    else:
        print("📡 DSSC Realtime Observations: DISABLED")

    station_id = CONFIG["locations"]["davos"]["station_id"]
    lat = CONFIG["locations"]["davos"]["lat"]
    lon = CONFIG["locations"]["davos"]["lon"]

    today = datetime.now()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2)]

    df_mosmix = fetch_mosmix(station_id)
    df_om = fetch_openmeteo(lat, lon, dates[0], dates[-1])

    if df_mosmix is None or df_om is None:
        print("❌ Failed to retrieve required forecast data.")
        sys.exit(1)

    df_combined = df_mosmix.join(df_om, how="inner")

    predictor = StandaloneWindPredictor(EXPORTED_WEIGHTS)
    df_predicted = predictor.predict(df_combined)

    days_data = {}
    for d_str in dates:
        df_day = df_predicted[df_predicted.index.strftime("%Y-%m-%d") == d_str]
        if not df_day.empty:
            dssc_hourly = None
            if args.include_dssc:
                dssc_wind = fetch_dssc_data("winddata")
                dssc_wind_dir = fetch_dssc_data("wdirdata")
                dssc_temp = fetch_dssc_data("tempdata")
                dssc_hourly = process_dssc_hourly(dssc_wind, dssc_wind_dir, dssc_temp, d_str)

            graph_name = f"graph_{d_str}.png"
            generate_day_graph(d_str, df_day, dssc_hourly, graph_name)
            days_data[d_str] = {
                "df": df_day,
                "dssc": dssc_hourly,  # Fixed variable reference
                "graph_name": graph_name
            }

    generate_mobile_html(days_data, "index.html")

if __name__ == "__main__":
    main()
