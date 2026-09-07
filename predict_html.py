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
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt

from foehn_gradient import get_combined_data_foehn_gradient
from CategorizedWindCorrectionPipeline import CategorizedWindCorrectionPipeline
from ecowitt import get_ecowitt_data, extract_weather_timeseries

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
    kelvin_to_celsius,
    degrees_to_cardinal,
    clean_namespaces,
    fetch_ms_hourly_data,
    get_ms_hourly_for_date,
    fetch_ms_now_data,
    compute_ms_hourly_so_far,
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

# Open-Meteo response cache -- see fetch_openmeteo()'s docstring. A
# plain file next to the script (not a repo path, not committed) so it
# only lives for the duration of one runner/workflow execution; the
# short TTL means a stale file left over from a previous run (if the
# runner filesystem were ever reused) can't silently serve outdated
# forecast data.
OPENMETEO_CACHE_FILE = Path(".openmeteo_cache.json")
OPENMETEO_CACHE_TTL_SECONDS = 30 * 60  # comfortably covers the two
                                        # back-to-back invocations
                                        # (--dssc, then plain) the
                                        # workflow runs in one job


def _load_openmeteo_cache(cache_key):
    """Returns a cached Open-Meteo DataFrame for cache_key if the cache
    file exists, matches this exact key, and is within
    OPENMETEO_CACHE_TTL_SECONDS -- else None (any problem reading/
    parsing the cache is treated the same as "no cache", never as an
    error, since the cache is purely an optimization)."""
    if not OPENMETEO_CACHE_FILE.exists():
        return None
    try:
        with open(OPENMETEO_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("key") != cache_key:
            return None
        if time.time() - payload.get("fetched_at", 0) > OPENMETEO_CACHE_TTL_SECONDS:
            return None
        df = pd.DataFrame(payload["data"])
        df.index = pd.to_datetime(payload["index"])
        return df
    except Exception:
        return None


def _save_openmeteo_cache(cache_key, df):
    """Best-effort write of a successful Open-Meteo response to disk.
    Never raises -- a failure to cache should not fail the run that
    just successfully fetched live data."""
    try:
        payload = {
            "key": cache_key,
            "fetched_at": time.time(),
            "index": [ts.isoformat() for ts in df.index],
            "data": df.to_dict(orient="list"),
        }
        with open(OPENMETEO_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"⚠️ Warning: could not write Open-Meteo cache: {e}")


def fetch_openmeteo(lat, lon, start_date, end_date):
    """Fetches Open-Meteo hourly forecast fields used as candidate features
    in the fitted category feature sets (see model_weights.json /
    model_weights_dssc.json's per-category "features" list, loaded via
    CategorizedWindCorrectionPipeline.from_exported_weights).

    Column names and units are now identical to wingfoil_predictor.py's
    fetch_openmeteo (wind_speed_unit=kn, so no manual m/s->kt conversion;
    om_wind_speed_700hPa_kt / om_wind_direction_700hPa naming, plus the
    10m wind fields). Previously this function used different parameters
    (wind_speed_unit=ms + manual conversion) and different column names
    (om_syn_ff_kt, om_syn_dd_deg, om_syn_800hPa_kt) than the training
    script -- the fitted models' feature lists reference
    om_wind_speed_700hPa_kt / om_wind_direction_700hPa, so those features
    were silently absent (defaulted to 0.0 in the correction pipeline) for
    every live prediction made by this script, even though the
    corresponding model was fit on the real values.

    Resilience: the GitHub Actions workflow invokes this script twice,
    back-to-back, with identical (lat, lon, start_date, end_date) --
    once with --dssc, once without -- so a single transient Open-Meteo
    slowdown used to have two independent chances to blow past the
    request timeout and sys.exit(1) the whole run. Two things soften
    that: (1) the HTTP call itself now retries transient failures
    (timeouts, connection errors, 5xx) a few times with backoff instead
    of giving up after one attempt; (2) a successful response is cached
    to a small on-disk JSON file keyed on the exact parameters used, so
    the second invocation in the same workflow run reuses it instead of
    hitting the live API again at all.
    """
    cache_key = f"{lat}:{lon}:{start_date}:{end_date}"
    cached_df = _load_openmeteo_cache(cache_key)
    if cached_df is not None:
        print("♻️  Open-Meteo: reusing cached response from earlier this run.")
        return cached_df

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
    session = requests.Session()
    # 3 attempts total, waiting 2s/4s/8s between them, on connection
    # errors, read timeouts, and 429/5xx responses -- a lone transient
    # hiccup (slow edge node, brief throttling on a shared runner IP,
    # ...) now has a chance to clear before the run gives up on it.
    retry_cfg = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry_cfg))
    try:
        # (connect_timeout, read_timeout) -- read given more headroom
        # than the old flat 10s, since that's the leg that was timing
        # out in practice.
        res = session.get(url, params=params, timeout=(5, 20))
        if res.status_code == 200:
            h = res.json().get("hourly", {})
            df = pd.DataFrame({
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
            _save_openmeteo_cache(cache_key, df)
            return df
        else:
            print(f"⚠️ Open-Meteo returned HTTP {res.status_code}")
    except Exception as e:
        print(f"⚠️ Error fetching Open-Meteo: {e}")
    return None

# =====================================================================
# FETCH & PROCESS DSSC DATA
# =====================================================================
# DSSC's old Cumulus MX endpoint (dssc.ch/cumulusmx/*.json) was
# deactivated. DSSC's station is now published as a shared Ecowitt
# device, so fetching/parsing it reuses ecowitt.py's get_ecowitt_data()
# and extract_weather_timeseries() directly -- exactly like
# wingfoil_predictor.py's own fetch_dssc_data/process_dssc_hourly, so
# both scripts read DSSC the same way instead of predict_html.py
# quietly hitting a dead endpoint and always showing no DSSC data.
#
# IMPORTANT: cross-checked against ground-truth readings on 2026-09-05
# (a real 7.2/8.8 m/s observation at 08:10 came back as raw "7.2"/"8.8")
# -- for THIS device, the windspeedmph/windgustmph fields returned by
# get_data are actually m/s, and tempf is actually °C, despite the field
# names. ecowitt.py's extract_weather_timeseries() now converts speed
# from m/s to the requested speed_unit and passes temp through as-is, so
# DSSC just calls it with speed_unit="knots". If DSSC's account settings
# ever change again, re-run the ground-truth cross-check before assuming
# otherwise.
DSSC_DEFAULT_DEVICE_ID = "Mzk5bHJCSWxMREpWTEFtKzhoQ1lPUT09"
DSSC_DEFAULT_AUTHORIZE = "8E98BV"

_dssc_session = None


def _get_dssc_session():
    """Lazily creates/reuses a requests.Session with cookies initialised
    against DSSC's ecowitt.net share page, mirroring ecowitt.py's own
    session.get(init_url) warm-up in main() before calling get_ecowitt_data."""
    global _dssc_session
    if _dssc_session is None:
        _dssc_session = requests.Session()
        init_url = (
            f"https://www.ecowitt.net/home/share"
            f"?authorize={DSSC_DEFAULT_AUTHORIZE}&device_id={DSSC_DEFAULT_DEVICE_ID}"
        )
        init_headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
            )
        }
        try:
            _dssc_session.get(init_url, headers=init_headers, timeout=10)
        except Exception as e:
            print(f"[⚠️ Warning] Station DSSC (init session) indisponible : {e}")
    return _dssc_session


def fetch_dssc_data(target_date):
    """Fetches one day of DSSC observations via ecowitt.py's own
    get_ecowitt_data(), for DSSC_DEFAULT_DEVICE_ID/DSSC_DEFAULT_AUTHORIZE.

    target_date: 'YYYY-MM-DD' string. Returns the raw JSON dict from
    ecowitt.net, or None on failure -- get_ecowitt_data already prints
    its own warning and returns None on request errors, so callers
    degrade the same way a Cumulus MX outage used to (an empty/None
    dssc_hourly, not a crash).
    """
    session = _get_dssc_session()
    return get_ecowitt_data(session, target_date, DSSC_DEFAULT_DEVICE_ID, DSSC_DEFAULT_AUTHORIZE)


def process_dssc_hourly(json_data, target_date):
    """Buckets one day's DSSC payload (as returned by fetch_dssc_data)
    into per-hour speed/gust/dir/temp observations -- same
    {'HH:00': {...} | None} shape the old Cumulus MX-based version
    produced, so every downstream caller (generate_day_graph,
    generate_mobile_html, ...) is unaffected by the API swap.

    Wind/temperature parsing itself is entirely ecowitt.py's
    extract_weather_timeseries() -- speed_unit="knots" converts the
    device's native m/s readings to knots, temp is used as-is in °C
    (see note above DSSC_DEFAULT_DEVICE_ID). This function only adds
    DSSC's own concerns on top: filtering to target_date, bucketing by
    hour, and circular-averaging direction.
    """
    hourly_raw = {f"{h:02d}:00": {"speeds": [], "gusts": [], "dirs": [], "temps": []} for h in range(0, 24)}

    times, speeds, gusts, dirs, temps = extract_weather_timeseries(json_data, speed_unit="knots")

    for i, time_val in enumerate(times):
        try:
            if str(time_val).isdigit():
                dt = datetime.fromtimestamp(int(time_val), tz=timezone.utc).astimezone(ZoneInfo("Europe/Zurich")).replace(tzinfo=None)
            else:
                dt = datetime.strptime(str(time_val), "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            continue

        if dt.strftime("%Y-%m-%d") != target_date:
            continue
        time_str = dt.strftime("%H:00")
        if time_str not in hourly_raw:
            continue

        # extract_weather_timeseries() now returns None (not 0.0) for
        # any sample it couldn't parse -- skip those instead of feeding
        # None into the sum()/max() averaging below.
        if i < len(speeds) and speeds[i] is not None:
            hourly_raw[time_str]["speeds"].append(speeds[i])
        if i < len(gusts) and gusts[i] is not None:
            hourly_raw[time_str]["gusts"].append(gusts[i])
        if i < len(dirs) and dirs[i] is not None:
            hourly_raw[time_str]["dirs"].append(dirs[i])
        if i < len(temps) and temps[i] is not None:
            hourly_raw[time_str]["temps"].append(temps[i])

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
                "speed": (sum(data["speeds"]) / len(data["speeds"])) if data["speeds"] else None,
                "gust": max(data["gusts"]) if data["gusts"] else (max(data["speeds"]) if data["speeds"] else None),
                "dir": avg_dir,
                "temp": sum(data["temps"]) / len(data["temps"]) if data["temps"] else None
            }
        else:
            hourly_obs[hour] = None
    return hourly_obs


def build_dssc_now_df(json_data, target_date):
    """Builds a live 10-minute-resolution DataFrame (index = Europe/
    Zurich-local timestamp, columns dssc_speed_kt/dssc_gust_kt) from
    DSSC's own raw ecowitt payload -- i.e. the native ~10-minute-
    resolution points, not the hourly buckets process_dssc_hourly
    produces. Mirrors wingfoil_predictor.py's build_dssc_now_df/
    wf_common.fetch_ms_now_data shape, so generate_day_graph's live-
    curve block can treat either source interchangeably.

    Unlike MS, DSSC has no separate "hourly" vs "now" endpoint: the one
    ecowitt.net payload (json_data, from fetch_dssc_data) already carries
    the day's raw 10-minute samples, so this just reshapes it into a
    DataFrame sliced to target_date.

    Returns an empty DataFrame (not None) if there is nothing usable, so
    callers can use the same `.empty` check as ms_now_df.
    """
    times, speeds, gusts, dirs, _temps = extract_weather_timeseries(json_data, speed_unit="knots")

    rows = []
    for i, time_val in enumerate(times):
        try:
            if str(time_val).isdigit():
                dt = datetime.fromtimestamp(int(time_val), tz=timezone.utc).astimezone(ZoneInfo("Europe/Zurich")).replace(tzinfo=None)
            else:
                dt = datetime.strptime(str(time_val), "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            continue

        if dt.strftime("%Y-%m-%d") != target_date:
            continue

        # None (missing/unparseable) maps to NaN here -- this is the
        # DataFrame boundary, and NaN is what lets the downstream plot
        # show a gap instead of dropping to zero.
        speed_val = speeds[i] if i < len(speeds) else None
        gust_val = gusts[i] if i < len(gusts) else None
        rows.append({
            "datetime": dt,
            "dssc_speed_kt": speed_val if speed_val is not None else np.nan,
            "dssc_gust_kt": gust_val if gust_val is not None else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

# =====================================================================
# GRAPH GENERATION
# =====================================================================

def generate_day_graph(date_str, df_day, output_path, build_time_str=None, obs_hourly=None,
                        obs_label="MS", now_hours=None, now_speed=None, now_gust=None):
    """Renders one day's forecast graph, with hourly speed/gust observation
    markers from a SINGLE source (obs_label: "MS" or "DSSC" -- see main()'s
    --dssc handling), plus that same source's live 10-minute curve for
    today if provided. Unlike the old overlay version, this never mixes
    MS and DSSC observations on one plot -- each run (plain or --dssc)
    only ever has one ground-truth source in scope, matching
    wingfoil_predictor.py's plot_prediction_summary."""
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

    # Hourly observations -- the single active source (MS or DSSC),
    # filled circle/x markers (no more filled-vs-unfilled distinction
    # since only one source is ever shown on a given graph now).
    obs_sp, obs_gust_sp = [], []
    if obs_hourly:
        obs_speed_h = [
            int(k.split(":")[0])
            for k, v in obs_hourly.items()
            if v is not None and v.get("speed") is not None
        ]
        obs_sp = [
            v["speed"]
            for k, v in obs_hourly.items()
            if v is not None and v.get("speed") is not None
        ]
        if obs_speed_h:
            ax.scatter(obs_speed_h, obs_sp, color="#2ecc71", marker="o",
                       label=f"{obs_label} Speed", zorder=5, s=30)

        obs_gust_h = [
            int(k.split(":")[0])
            for k, v in obs_hourly.items()
            if v is not None and v.get("gust") is not None
        ]
        obs_gust_sp = [
            v["gust"]
            for k, v in obs_hourly.items()
            if v is not None and v.get("gust") is not None
        ]
        if obs_gust_h:
            ax.scatter(obs_gust_h, obs_gust_sp, color="#e67e22", marker="x",
                       label=f"{obs_label} Gust", zorder=5, s=30)

    # Live 10-minute "now" curve -- only ever populated for the current
    # day (see main()'s now_df handling), for whichever source
    # (MS/DSSC) this graph is showing. Drawn as thin, semi-transparent
    # lines (not scatter) at the same green/orange colors so it reads as
    # the fine-grained trace behind the same-colored hourly markers
    # above, rather than a third, competing series.
    if now_hours is not None and now_speed is not None and pd.Series(now_speed).notna().any():
        ax.plot(
            now_hours, now_speed,
            label=f"{obs_label} 10min Sp.", color="#2ecc71",
            linewidth=1.0, alpha=0.5, zorder=4
        )
    if now_hours is not None and now_gust is not None and pd.Series(now_gust).notna().any():
        ax.plot(
            now_hours, now_gust,
            label=f"{obs_label} 10min Gust", color="#e67e22",
            linewidth=1.0, alpha=0.5, zorder=4
        )

    ax.axhline(CONFIG["settings"]["wind_threshold_knots"], color="#e74c3c", linestyle=":", alpha=0.7, label="Threshold (10kt)")
    
    # Limits & Spacing
    ax.set_xlim(10, 19)
    ax.set_xticks(range(10, 20))

    # Y-axis upper bound: 25kt by default, grown in 5kt increments only
    # when the data actually needs the extra headroom (raw/corrected wind
    # & gust and the active source's observations -- the ±1 std
    # uncertainty bands are deliberately NOT considered here, so a
    # wide-but-low-confidence band doesn't by itself push the axis
    # taller), rather than clipping tall days at a fixed 25kt ceiling.
    DEFAULT_Y_MAX = 25
    Y_STEP = 5.0
    
    mask = (hours >= 10) & (hours <= 19)
    candidate_maxes = [
        df_day.loc[mask, "mosmix_ff_kt"].max(skipna=True),
        df_day.loc[mask, "mosmix_fx1_kt"].max(skipna=True),
        corr_ff[mask].max(skipna=True),
        corr_fx[mask].max(skipna=True),
    ]

    if obs_sp:
        candidate_maxes.append(max(obs_sp))
    if obs_gust_sp:
        candidate_maxes.append(max(obs_gust_sp))
    if now_speed is not None and pd.Series(now_speed).notna().any():
        candidate_maxes.append(pd.Series(now_speed).max(skipna=True))
    if now_gust is not None and pd.Series(now_gust).notna().any():
        candidate_maxes.append(pd.Series(now_gust).max(skipna=True))

    candidate_maxes = [v for v in candidate_maxes if pd.notna(v)]
    data_max = max(candidate_maxes) if candidate_maxes else 0.0

    if data_max > DEFAULT_Y_MAX:
        y_max = int(np.ceil(data_max / Y_STEP) * Y_STEP)
    else:
        y_max = DEFAULT_Y_MAX

    ax.set_ylim(0, y_max)

    # 1-knot grid ticks on the y-axis
    y_ticks = np.arange(0, y_max + 1, 1)
    y_labels = [str(y) if y % 1 == 0 else "" for y in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis='y', right=True, labelright=True)

    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
    date_with_weekday = dt_obj.strftime("%A, %Y-%m-%d")

    ax.set_xlabel("Local Hour")
    ax.set_ylabel("Wind Speed (knots)")
    ax.set_title(f"Davosersee Forecast — {date_with_weekday} ({obs_label})", fontsize=11, fontweight="bold")
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

def generate_mobile_html(days_data, output_file="index.html", source_label="MS", sidecar_path=None):
    """Renders index.html from days_data (this run's rows), persists this
    run's data to sidecar_path so a later run of the OTHER source can pick
    it back up, and loads that other source's sidecar (if present) so a
    single index.html always contains both MS and DSSC data/graphs, with a
    client-side toggle (?src=ms or ?src=dssc in the URL, defaulting to
    MS) switching which one is visible -- since each invocation of this
    script only computes one source (MS by default, or DSSC via --dssc),
    neither run alone has both to render.

    The MS/DSSC toggle buttons themselves are hidden by default and are
    only revealed when the page is loaded with ?dssc=1 in the URL --
    this is a display-only gate (the static file always contains both
    sources' data/markup either way), enforced client-side in the
    <script> block below since this HTML is generated once and then
    served as-is, so there's no per-request server logic to gate it at.
    """
    version_str, weights_updated, build_time_str = get_formatted_version_and_build()

    # Persist this run's rows (JSON-serializable subset only -- see
    # _serialize_days_data) so the other source's next run can merge it
    # back in without needing to recompute anything.
    serializable = _serialize_days_data(days_data)
    if sidecar_path:
        try:
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({
                    "source_label": source_label,
                    "build_time_str": build_time_str,
                    "version_str": version_str,
                    "weights_updated": weights_updated,
                    "days": serializable,
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: could not write sidecar {sidecar_path}: {e}")

    # Load the OTHER source's most recent sidecar, if any, so this page
    # still shows something for the toggle target even though this run
    # only just computed `source_label`.
    other_label = "DSSC" if source_label == "MS" else "MS"
    other_sidecar_path = SIDECAR_PATHS.get(other_label)
    other_payload = None
    if other_sidecar_path and Path(other_sidecar_path).exists():
        try:
            with open(other_sidecar_path, "r", encoding="utf-8") as f:
                other_payload = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: could not read sidecar {other_sidecar_path}: {e}")

    sources = {source_label: serializable}
    build_times = {source_label: build_time_str}
    versions = {source_label: version_str}
    weights_updates = {source_label: weights_updated}
    if other_payload:
        sources[other_label] = other_payload.get("days", {})
        build_times[other_label] = other_payload.get("build_time_str", "Unknown")
        versions[other_label] = other_payload.get("version_str", "Unknown")
        weights_updates[other_label] = other_payload.get("weights_updated", "Unknown")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Davosersee Wind Forecast</title>
    <style>
        html, body {{ height: 100%; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 0; color: #212529; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
        .header {{ background: #1e293b; color: white; padding: 14px 12px; margin: 0; flex: 0 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.15); z-index: 10; }}
        .header h1 {{ margin: 0; font-size: 1.2rem; }}
        .version {{ font-size: 0.75rem; color: #94a3b8; margin-top: 4px; line-height: 1.4; }}
        .source-toggle {{ margin-top: 8px; }}
        .source-toggle a {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; text-decoration: none; color: #cbd5e1; border: 1px solid #475569; margin-right: 6px; }}
        .source-toggle a.active {{ background: #38bdf8; color: #0f172a; border-color: #38bdf8; font-weight: 600; }}
        /* Scrollable content area below the fixed header. Both the MS and
           DSSC .source-section divs live inside THIS one shared scroller
           (see the markup below) rather than each having their own -- that
           is what makes switching sources preserve scroll position: the
           scrollTop being saved/restored belongs to #scroll-frame, not to
           whichever section happens to be visible, so there is exactly one
           scroll position to carry across the toggle. */
        #scroll-frame {{ flex: 1 1 auto; overflow-y: auto; -webkit-overflow-scrolling: touch; padding: 12px; box-sizing: border-box; }}
        .day-card {{ background: white; border-radius: 10px; padding: 12px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow-x: auto; }}
        .day-title {{ font-weight: bold; font-size: 1.1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; color: white; }}
        .bg-go {{ background: #22c55e; }}
        .bg-nogo {{ background: #ef4444; }}
        img {{ width: 100%; border-radius: 6px; margin: 8px 0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.72rem; margin-top: 8px; white-space: nowrap; }}
        th, td {{ padding: 6px 4px; text-align: center; border-bottom: 1px solid #f1f5f9; }}
        th {{ background: #f8fafc; color: #64748b; font-weight: 600; }}
        .source-section {{ display: none; }}
        .source-section.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏄 Davosersee Wind Forecast by Camille, DWD MOSMIX-L data with corrections</h1>
        <h1>-- experimental, at your own risk, no guarantees! --</h1>
        <div class="version" id="version-info"></div>
        <div class="source-toggle" id="source-toggle" style="display: none;">
            <a href="?src=ms&dssc=1" data-src="MS">MS (MeteoSwiss)</a><a href="?src=dssc&dssc=1" data-src="DSSC">DSSC</a>
        </div>
    </div>
    <div id="scroll-frame">
"""

    version_info_by_source = {}
    for label in sources.keys():
        version_info_by_source[label] = (
            f"Build Time: {build_times.get(label, 'Unknown')}<br>"
            f"Model Weights Version: {versions.get(label, 'Unknown')} "
            f"(exported {weights_updates.get(label, 'Unknown')}) "
            f"— Source: {label}"
        )

    for label, days in sources.items():
        html_content += f"""
    <div class="source-section" data-src="{label}">
"""
        for date_str, day in days.items():
            status_badge = (
                '<span class="badge bg-go">🟢 WIN*FOIL</span>' if day.get("go")
                else '<span class="badge bg-nogo">🔴 avg.Wind<10kt </span>'
            )
            html_content += f"""
        <div class="day-card">
            <div class="day-title">
                <span>{date_str}</span>
                {status_badge}
            </div>
            <img src="{day['graph_name']}" alt="Forecast Graph">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Wind Raw (kt)</th>
                        <th>Wind Corr (kt)</th>
                        <th>Gust Corr (kt)</th>
                        <th>Wind Dir</th>
                        <th>Temp (°C)</th>
                        <th>Rain Prob (%)</th>
                        <th>Cloud (%)</th>
                        <th>BL (m)</th>
                        <th>Foehn Grad (hPa)</th>
                        <th>{label} Spd (kt)</th>
                        <th>{label} Gust (kt)</th>
                        <th>Regime</th>
                    </tr>
                </thead>
                <tbody>
"""
            for row in day["rows"]:
                html_content += f"""
                    <tr>
                        <td>{row['time']}</td>
                        <td>{row['raw_ff']}</td>
                        <td><b>{row['corr_ff']}</b></td>
                        <td>{row['corr_fx']}</td>
                        <td>{row['wind_dir']}</td>
                        <td>{row['temp']}</td>
                        <td>{row['rain']}</td>
                        <td>{row['cloud']}</td>
                        <td>{row['bl_height']}</td>
                        <td>{row['foehn_grad']}</td>
                        <td>{row['obs_speed']}</td>
                        <td>{row['obs_gust']}</td>
                        <td>{row['classification']}</td>
                    </tr>"""

            html_content += """
                </tbody>
            </table>
        </div>"""

        html_content += """
    </div>"""

    html_content += """
    </div>
"""

    version_info_json = json.dumps(version_info_by_source)

    html_content += f"""
    <script>
        const versionInfo = {version_info_json};
        const params = new URLSearchParams(window.location.search);
        let src = (params.get('src') || 'ms').toLowerCase();
        let activeLabel = src === 'dssc' ? 'DSSC' : 'MS';
        const scrollFrame = document.getElementById('scroll-frame');

        // The MS/DSSC toggle buttons are opt-in: only shown when the
        // page is loaded with ?dssc=1, so casual visitors don't see a
        // switch for a data source they haven't asked to see.
        const toggleEl = document.getElementById('source-toggle');
        const dsscParam = params.get('dssc') === '1';
        if (toggleEl) {{
            toggleEl.style.display = dsscParam ? '' : 'none';
        }}

        function applyActiveSource(label) {{
            document.querySelectorAll('.source-section').forEach(el => {{
                el.classList.toggle('active', el.getAttribute('data-src') === label);
            }});
            document.querySelectorAll('.source-toggle a').forEach(el => {{
                el.classList.toggle('active', el.getAttribute('data-src') === label);
            }});
            const versionEl = document.getElementById('version-info');
            if (versionEl) {{
                versionEl.innerHTML = versionInfo[label] || versionInfo['MS'] || '';
            }}
        }}

        applyActiveSource(activeLabel);

        // Switching MS<->DSSC is handled entirely client-side (no page
        // reload) so the SAME #scroll-frame element -- shared by both
        // .source-section divs -- keeps whatever scrollTop the visitor
        // was already at. That is what makes the two views directly
        // comparable: scrolling to, say, tomorrow's graph in MS and then
        // switching to DSSC lands on tomorrow's graph in DSSC too,
        // instead of resetting to the top the way a normal link
        // navigation (or independently-scrolling sections) would.
        if (toggleEl) {{
            toggleEl.querySelectorAll('a').forEach(el => {{
                el.addEventListener('click', (evt) => {{
                    evt.preventDefault();
                    const label = el.getAttribute('data-src');
                    if (label === activeLabel) return;
                    activeLabel = label;
                    applyActiveSource(activeLabel);
                    const newSrc = label === 'DSSC' ? 'dssc' : 'ms';
                    const url = new URL(window.location.href);
                    url.searchParams.set('src', newSrc);
                    url.searchParams.set('dssc', '1');
                    window.history.replaceState({{}}, '', url);
                    // scrollFrame.scrollTop is untouched by the section
                    // toggle above (display:none/block on a child doesn't
                    // move its scrolled ancestor), so no explicit
                    // save/restore is even needed here -- the frame simply
                    // never moved. Left as a no-op comment rather than
                    // silently relying on that being obvious to a future
                    // reader.
                }});
            }});
        }}
    </script>
</body>
</html>"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"📱 Mobile HTML dashboard generated: {output_file}")


def _serialize_days_data(days_data):
    """Reduces main()'s in-memory days_data (DataFrames + hourly obs
    dicts) to a plain, JSON-serializable {date: {...}} structure --
    exactly the fields generate_mobile_html's table/badge rendering
    needs -- so it can be written to a sidecar file and read back by a
    later run of the OTHER source (see generate_mobile_html)."""
    out = {}
    for date_str, data in days_data.items():
        df = data["df"]
        obs_hourly = data.get("obs_hourly") or {}
        go = False
        rows = []
        for ts, row in df.iterrows():
            if not (10 <= ts.hour <= 19):
                continue
            if pd.notna(row.get("mosmix_ff_corrected_kt")) and row["mosmix_ff_corrected_kt"] >= 10:
                go = True

            obs_hour_obs = obs_hourly.get(ts.strftime("%H:00"))
            obs_speed = f"{obs_hour_obs['speed']:.1f}" if obs_hour_obs and obs_hour_obs.get('speed') is not None else "-"
            obs_gust = f"{obs_hour_obs['gust']:.1f}" if obs_hour_obs and obs_hour_obs.get('gust') is not None else "-"

            rows.append({
                "time": ts.strftime("%H:%M"),
                "raw_ff": f"{row['mosmix_ff_kt']:.1f}" if pd.notna(row.get('mosmix_ff_kt')) else "-",
                "corr_ff": f"{row['mosmix_ff_corrected_kt']:.1f}" if pd.notna(row.get('mosmix_ff_corrected_kt')) else "-",
                "corr_fx": f"{row['mosmix_fx1_corrected_kt']:.1f}" if pd.notna(row.get('mosmix_fx1_corrected_kt')) else "-",
                "wind_dir": degrees_to_cardinal(row.get('mosmix_dd_deg')),
                "temp": f"{row['mosmix_tt_c']:.1f}" if pd.notna(row.get('mosmix_tt_c')) else "-",
                "cloud": f"{row['mosmix_cloud_pct']:.0f}%" if pd.notna(row.get('mosmix_cloud_pct')) else "-",
                "bl_height": f"{row['om_bl_height']:.0f}" if pd.notna(row.get('om_bl_height')) else "-",
                "rain": f"{row['om_prec_prob']:.0f}%" if pd.notna(row.get('om_prec_prob')) else "-",
                "foehn_grad": f"{row['mosmix_dp_foehn']:.1f}" if pd.notna(row.get('mosmix_dp_foehn')) else "-",
                "obs_speed": obs_speed,
                "obs_gust": obs_gust,
                "classification": row.get('classification', '-'),
            })

        out[date_str] = {
            "graph_name": data["graph_name"],
            "go": go,
            "rows": rows,
        }
    return out

# =====================================================================
# MAIN EXECUTION ROUTINE
# =====================================================================

# Sidecar files each run's (MS or DSSC) table/graph data is persisted to,
# so generate_mobile_html can merge in whichever source THIS run didn't
# just compute -- see that function's docstring.
SIDECAR_PATHS = {
    "MS": "days_data_ms.json",
    "DSSC": "days_data_dssc.json",
}

def main():
    parser = argparse.ArgumentParser(description="Standalone Wingfoil Predictor & Dashboard Generator")
    parser.add_argument("--dssc", action="store_true",
                         help="Use DSSC as the observation source (DSSC-fitted weights, DSSC hourly/live obs, "
                              "_dssc-suffixed plot files) instead of MS")
    parser.add_argument("--weights-file", type=str, default=None,
                         help="Path to weights JSON file (default: model_weights_dssc.json with --dssc, "
                              "else model_weights.json)")
    args = parser.parse_args()

    source_label = "DSSC" if args.dssc else "MS"
    weights_file = args.weights_file or ("model_weights_dssc.json" if args.dssc else "model_weights.json")

    global EXPORTED_WEIGHTS
    EXPORTED_WEIGHTS = load_exported_weights(weights_file)
    version_str, weights_updated, build_time_str = get_formatted_version_and_build()

    print(f"🚀 Running Wingfoil Prediction Engine [{version_str} - Exported: {weights_updated}]")
    print(f"🕒 Build Time: {build_time_str}")
    print(f"📡 Observation source for this run: {source_label} (weights: {weights_file})")

    station_id = CONFIG["locations"]["davos"]["station_id"]
    lat = CONFIG["locations"]["davos"]["lat"]
    lon = CONFIG["locations"]["davos"]["lon"]
    ms_station_abbr = CONFIG["locations"]["davos"].get("ms_station_abbr", "DAV")

    tz_name = CONFIG["settings"]["timezone"]
    today = datetime.now(ZoneInfo(tz_name))
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)]
    # Graph filenames get a _dssc suffix for a --dssc run, so an MS run
    # and a DSSC run never overwrite each other's plot files -- index.html
    # references whichever set matches the source the visitor toggled to.
    base_img_names = ["today", "tomorrow", "dayaftertomorrow"]
    suffix = "_dssc" if args.dssc else ""
    img_names = [f"{name}{suffix}.png" for name in base_img_names]

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

    # Only the active source (MS by default, or DSSC via --dssc) is
    # fetched and shown -- matching wingfoil_predictor.py's analyze_day/
    # plot_prediction_summary gating, so a run's graph/table never mixes
    # observation sources.
    ms_df = None
    if not args.dssc:
        print(f"📥 Récupération des observations MS (MeteoSwiss, station {ms_station_abbr})...")
        ms_df = fetch_ms_hourly_data(station_abbr=ms_station_abbr)
        if ms_df is None or ms_df.empty:
            print("⚠️ Warning: MS data unavailable -- graphs/table will show no observations for this run.")

    for i, d_str in enumerate(dates):
        df_day = df_predicted[df_predicted.index.strftime("%Y-%m-%d") == d_str]
        if not df_day.empty:
            now_hours = now_speed = now_gust = None
            is_today = (d_str == today.strftime("%Y-%m-%d"))

            if args.dssc:
                dssc_json = fetch_dssc_data(d_str)
                obs_hourly = process_dssc_hourly(dssc_json, d_str)

                # DSSC's own live 10-minute curve for today, built from
                # the same dssc_json payload already fetched above --
                # mirrors wingfoil_predictor.py's build_dssc_now_df.
                if is_today:
                    now_df = build_dssc_now_df(dssc_json, d_str)
                    if not now_df.empty:
                        now_hours = now_df.index.hour + now_df.index.minute / 60.0
                        now_speed = now_df["dssc_speed_kt"] if "dssc_speed_kt" in now_df else None
                        now_gust = now_df["dssc_gust_kt"] if "dssc_gust_kt" in now_df else None
            else:
                obs_hourly = get_ms_hourly_for_date(ms_df, d_str)

                # For TODAY only, additionally pull the MS 10-minute
                # "_t_now_" file (same as wingfoil_predictor.py's
                # main()): it gives a running average for the current,
                # still-incomplete hour (folded into obs_hourly so the
                # table's current-hour row and the "GO" badge see it
                # too) plus the raw 10-minute points used for the
                # finer-grained "now" curve on the graph. Past days are
                # untouched -- now_hours/speed/gust stay None.
                if is_today:
                    print("📥 Récupération des observations MS 10 min (station DAV, jour courant)...")
                    ms_now_df = fetch_ms_now_data(station_abbr=ms_station_abbr)
                    ms_hourly_so_far = compute_ms_hourly_so_far(ms_now_df, d_str)
                    for hour_str, obs in ms_hourly_so_far.items():
                        obs_hourly[hour_str] = obs

                    if ms_now_df is not None and not ms_now_df.empty:
                        day_mask = ms_now_df.index.strftime("%Y-%m-%d") == d_str
                        ms_now_day = ms_now_df[day_mask]
                        if not ms_now_day.empty:
                            now_hours = ms_now_day.index.hour + ms_now_day.index.minute / 60.0
                            now_speed = ms_now_day["ms_speed_kt"] if "ms_speed_kt" in ms_now_day else None
                            now_gust = ms_now_day["ms_gust_kt"] if "ms_gust_kt" in ms_now_day else None

            graph_name = img_names[i]
            generate_day_graph(
                d_str, df_day, graph_name, build_time_str=build_time_str,
                obs_hourly=obs_hourly, obs_label=source_label,
                now_hours=now_hours, now_speed=now_speed, now_gust=now_gust
            )
            days_data[d_str] = {
                "df": df_day,
                "obs_hourly": obs_hourly,
                "graph_name": graph_name
            }

    generate_mobile_html(
        days_data, "index.html",
        source_label=source_label, sidecar_path=SIDECAR_PATHS[source_label]
    )

if __name__ == "__main__":
    main()
