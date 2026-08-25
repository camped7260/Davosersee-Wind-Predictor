#!/usr/bin/env python3
"""
predict_html.py

Standalone prediction & HTML dashboard generator for Wingfoil forecasting (Davos).
Runs predictions for today and tomorrow by loading
weights, features, scalers, and offsets dynamically from `model_weights.json`.

Live prediction now runs through the exact same
CategorizedWindCorrectionPipeline (feature engineering, classification,
rain damping, Bayesian Ridge correction) that wingfoil_predictor.py trains
-- reconstructed from model_weights.json via
CategorizedWindCorrectionPipeline.from_exported_weights instead of the
hand-written StandaloneWindPredictor this script used to carry. See that
classmethod's docstring for why this matters: it's what makes live
prediction and training/analysis structurally unable to diverge in how a
forecast is classified or corrected.
"""

import argparse
import io
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from foehn_gradient import get_combined_data_foehn_gradient
from CategorizedWindCorrectionPipeline import CategorizedWindCorrectionPipeline

# Config loading, unit conversions, WMO code descriptions, and display
# constants (REGIME_COLORS etc.) come from wf_common.py -- the same shared
# foundation wingfoil_predictor.py uses. This script previously kept its
# own DEFAULT_CONFIG (with unused keys like init_valley_angle /
# bl_height_threshold_m, artifacts of StandaloneWindPredictor's now-removed
# hand-reimplementation) and never actually read config.json for most
# settings; using load_config() directly means config.json is genuinely
# the single on-disk source of truth for both scripts.
from wf_common import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    REGIME_COLORS,
    DEFAULT_REGIME_COLOR,
    describe_weather_code,
    load_config,
    convert_ms_to_knots,
    convert_kmh_to_knots,
    kelvin_to_celsius,
    degrees_to_cardinal,
    clean_namespaces,
)

# =====================================================================
# CONFIGURATION & WEIGHTS LOADING
# =====================================================================

CONFIG = load_config()

# Fallback weights in case weights.json is missing
DEFAULT_WEIGHTS = {
    "version": "v2.5-json",
    "updated_at": "2026-08-05 00:00:00 UTC",
    "global_fallback_bias": 0.45,
    "global_fx1_fallback_bias": 1.20,
    "category_mean_offsets": {
        "Sunny": 0.52,
        "PartlyCloudy": 0.38,
        "NordFoehn + Sunny": -1.15,
        "NordFoehn + PartlyCloudy": -0.85,
        "Cloudy": 0.12
    },
    "category_fx1_mean_offsets": {
        "Sunny": 1.10,
        "PartlyCloudy": 0.95,
        "NordFoehn + Sunny": -0.40,
        "NordFoehn + PartlyCloudy": -0.20,
        "Cloudy": 0.50
    },
    "bayesian_models": {},
    "bayesian_fx1_models": {}
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

def get_formatted_version_and_build():
    """Returns the weights export timestamp and local HTML build timestamp."""
    weights_updated = EXPORTED_WEIGHTS.get("updated_at", "Unknown")
    version_str = EXPORTED_WEIGHTS.get("version", "v2.5-json")

    tz_name = CONFIG["settings"]["timezone"]
    local_now = datetime.now(ZoneInfo(tz_name))
    build_time_str = local_now.strftime("%Y-%m-%d %H:%M:%S %Z")

    return version_str, weights_updated, build_time_str

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
    """Fetches Open-Meteo hourly forecast fields used by
    category_feature_sets / category_fx1_feature_sets.

    Column names and units are now identical to wingfoil_predictor.py's
    fetch_openmeteo (wind_speed_unit=kn, so no manual m/s->kt conversion;
    om_wind_speed_700hPa_kt / om_wind_direction_700hPa naming, plus the
    10m wind fields). Previously this function used different parameters
    (wind_speed_unit=ms + manual conversion) and different column names
    (om_syn_ff_kt, om_syn_dd_deg, om_syn_800hPa_kt) than the training
    script -- config.json's category_feature_sets reference
    om_wind_speed_700hPa_kt / om_wind_direction_700hPa, so those features
    were silently absent (defaulted to 0.0 in the correction pipeline) for
    every live prediction made by this script, even though the
    corresponding model was fit on the real values.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_700hPa,wind_direction_700hPa,weather_code,precipitation_probability,"
                  "boundary_layer_height,wind_speed_800hPa,wind_direction_800hPa,soil_temperature_0cm,"
                  "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kn",
        "timezone": "Europe/Zurich",
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            h = res.json().get("hourly", {})
            return pd.DataFrame({
                "om_wind_speed_700hPa_kt": h.get("wind_speed_700hPa", []),
                "om_wind_direction_700hPa": h.get("wind_direction_700hPa", []),
                "om_wind_speed_800hPa_kt": h.get("wind_speed_800hPa", []),
                "om_wind_direction_800hPa": h.get("wind_direction_800hPa", []),
                "om_wind_speed_10m_kt": h.get("wind_speed_10m", []),
                "om_wind_direction_10m": h.get("wind_direction_10m", []),
                "om_prec_prob": h.get("precipitation_probability", []),
                "om_w_codes": h.get("weather_code", []),
                "om_bl_height": h.get("boundary_layer_height", []),
                "om_soil_temp_0cm": h.get("soil_temperature_0cm", []),
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

def generate_day_graph(date_str, df_day, dssc_obs, output_path, build_time_str=None):
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
    ax.plot(hours, df_day["mosmix_fx1_kt"], label="Gust Raw (MOSMIX-L)", color="#f0b27a", linestyle="--", linewidth=1.2)
    ax.plot(hours, corr_ff, label="Wind Corrected", color="#2ecc71", linewidth=2.2)
    ax.plot(hours, corr_fx, label="Gust Corrected", color="#e67e22", linewidth=2.2)

    # ±1 std dev uncertainty bands
    if "mosmix_ff_std_kt" in df_day.columns:
        std_ff = df_day["mosmix_ff_std_kt"].fillna(0.0).values
        ax.fill_between(
            hours, np.maximum(0, corr_ff - std_ff), corr_ff + std_ff,
            color="#2ecc71", alpha=0.15, zorder=1
        )

    if "mosmix_fx1_std_kt" in df_day.columns:
        std_fx = df_day["mosmix_fx1_std_kt"].fillna(0.0).values
        ax.fill_between(
            hours, np.maximum(0, corr_fx - std_fx), corr_fx + std_fx,
            color="#e67e22", alpha=0.15, zorder=1
        )

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
            ax.scatter(obs_h, obs_sp, color="#2ecc71", label="Obs. Avg. Speed", zorder=5, s=30)

        # DSSC Gust Points
        gust_h = [
            int(k.split(":")[0]) 
            for k, v in dssc_obs.items() 
            if v is not None and v.get("gust") is not None
        ]
        gust_sp = [
            v["gust"] 
            for k, v in dssc_obs.items() 
            if v is not None and v.get("gust") is not None
        ]
        if gust_h:
            ax.scatter(gust_h, gust_sp, color="#e67e22", marker="2", label="Obs. Gust", zorder=5, s=30)

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
    ax.tick_params(axis='y', right=True, labelright=True)

    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_with_weekday = dt_obj.strftime("%A, %Y-%m-%d")

    ax.set_xlabel("Local Hour")
    ax.set_ylabel("Wind Speed (knots)")
    ax.set_title(f"Davosersee Forecast — {date_with_weekday}", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    if build_time_str:
        fig.text(
            0.98, 0.01, 
            f"Generated: {build_time_str}", 
            fontsize=6, 
            color="#7f8c8d", 
            ha="right", 
            va="bottom", 
            style="italic"
        )
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =====================================================================
# HTML DASHBOARD GENERATOR
# =====================================================================

def generate_mobile_html(days_data, output_file="index.html"):
    version_str, weights_updated, build_time_str = get_formatted_version_and_build()

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
        .version {{ font-size: 0.75rem; color: #94a3b8; margin-top: 4px; line-height: 1.4; }}
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
        <div class="version">
            Build Time: {build_time_str}<br>
            Model Weights Version: {version_str} (exported {weights_updated})
        </div>
    </div>
"""

    for date_str, data in days_data.items():
        status_badge = '<span class="badge bg-nogo">🔴 avg.Wind<10kt </span>'
        for ts, row in data["df"].iterrows():
            if 10 <= ts.hour <= 19 and row["mosmix_ff_corrected_kt"] >= 10:
                status_badge = '<span class="badge bg-go">🟢 WIN*FOIL</span>'
         
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
                    <th>BL (m)</th>
                    <th>Rain Prob (%)</th>
                    <th>Weather</th>
                    <th>Nordfoehn Grad (hPa)</th>
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
                bl_height = f"{row['om_bl_height']:.0f}" if pd.notna(row.get('om_bl_height')) else "-"
                rain = f"{row['om_prec_prob']:.0f}%" if pd.notna(row.get('om_prec_prob')) else "-"
                wcode_label = describe_weather_code(row.get('om_w_codes'))
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
                    <td>{bl_height}</td>
                    <td>{rain}</td>
                    <td>{wcode_label}</td>
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

    global EXPORTED_WEIGHTS
    EXPORTED_WEIGHTS = load_exported_weights(args.weights_file)
    version_str, weights_updated, build_time_str = get_formatted_version_and_build()

    print(f"🚀 Running Wingfoil Prediction Engine [{version_str} - Exported: {weights_updated}]")
    print(f"🕒 Build Time: {build_time_str}")
    if args.include_dssc:
        print("📡 DSSC Realtime Observations: ENABLED")
    else:
        print("📡 DSSC Realtime Observations: DISABLED")

    station_id = CONFIG["locations"]["davos"]["station_id"]
    lat = CONFIG["locations"]["davos"]["lat"]
    lon = CONFIG["locations"]["davos"]["lon"]

    tz_name = CONFIG["settings"]["timezone"]
    today = datetime.now(ZoneInfo(tz_name))
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]
    img_names = ["today.png", "tomorrow.png", "dayaftertomorrow.png"]

    df_mosmix = fetch_mosmix(station_id)
    df_om = fetch_openmeteo(lat, lon, dates[0], dates[-1])

    if df_mosmix is None or df_om is None:
        print("❌ Failed to retrieve required forecast data.")
        sys.exit(1)

    df_combined = df_mosmix.join(df_om, how="inner")

    # Foehn pressure gradient (mosmix_dp_foehn): fetched here, once, using
    # the same config.json "foehn_stations" (or foehn_gradient.py's
    # DEFAULT_STATIONS fallback) as wingfoil_predictor.py's main(), rather
    # than inside the predictor class. CategorizedWindCorrectionPipeline's
    # _prepare_features expects mosmix_dp_foehn to already be a column on
    # the input df -- it's the caller's responsibility to supply it,
    # exactly as wingfoil_predictor.py does -- so both scripts source this
    # feature identically instead of predict_html.py silently falling back
    # to a proxy (mosmix_u_kt * 0.4) whenever the fetch failed.
    print("📥 Récupération du gradient de Foehn...")
    foehn_stations = CONFIG.get("foehn_stations")
    unique_dates = sorted(df_combined.index.strftime("%Y-%m-%d").unique())
    foehn_records = []
    for target_date in unique_dates:
        records = get_combined_data_foehn_gradient(target_date, stations=foehn_stations)
        if records:
            foehn_records.extend(records)

    if foehn_records:
        df_foehn = (
            pd.DataFrame(foehn_records)
            .drop_duplicates(subset=["datetime"], keep="first")
            .set_index("datetime")
        )
        df_combined["mosmix_dp_foehn"] = df_combined.index.map(df_foehn["dp_foehn"])
    else:
        df_combined["mosmix_dp_foehn"] = np.nan

    if df_combined["mosmix_dp_foehn"].isna().any():
        missing = int(df_combined["mosmix_dp_foehn"].isna().sum())
        print(f"⚠️ Warning: mosmix_dp_foehn missing for {missing} row(s); falling back to proxy (mosmix_u_kt * 0.4) for those rows.")
        df_combined["mosmix_dp_foehn"] = df_combined["mosmix_dp_foehn"].fillna(
            df_combined["mosmix_u_kt"].apply(lambda u: u * 0.4 if pd.notna(u) else 0.0)
        )

    # Reconstruct the exact fit-time pipeline (feature engineering,
    # classification, rain damping, resolve_category fallbacks, and the
    # Bayesian Ridge correction itself) from the exported weights, instead
    # of predict_html.py's former hand-written StandaloneWindPredictor.
    # This is what guarantees live prediction and training/analysis
    # (wingfoil_predictor.py) apply identical logic -- see
    # CategorizedWindCorrectionPipeline.from_exported_weights.
    pipeline = CategorizedWindCorrectionPipeline.from_exported_weights(EXPORTED_WEIGHTS)
    df_predicted = pipeline.process(df_combined)

    days_data = {}
    for i, d_str in enumerate(dates):
        df_day = df_predicted[df_predicted.index.strftime("%Y-%m-%d") == d_str]
        if not df_day.empty:
            dssc_hourly = None
            if args.include_dssc:
                dssc_wind = fetch_dssc_data("winddata")
                dssc_wind_dir = fetch_dssc_data("wdirdata")
                dssc_temp = fetch_dssc_data("tempdata")
                dssc_hourly = process_dssc_hourly(dssc_wind, dssc_wind_dir, dssc_temp, d_str)

            graph_name = img_names[i]
            generate_day_graph(d_str, df_day, dssc_hourly, graph_name, build_time_str=build_time_str)
            days_data[d_str] = {
                "df": df_day,
                "dssc": dssc_hourly,
                "graph_name": graph_name
            }

    generate_mobile_html(days_data, "index.html")

if __name__ == "__main__":
    main()
