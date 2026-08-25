#!/usr/bin/env python3
"""
wf_common.py

Single shared foundation for the Wingfoil forecasting toolchain
(wingfoil_predictor.py, predict_html.py, CategorizedWindCorrectionPipeline.py,
foehn_gradient.py).

This module owns everything that previously existed as parallel, hand-copied
definitions in wingfoil_predictor.py and predict_html.py:
  - config.json loading (DEFAULT_CONFIG + load_config)
  - unit conversions (knots/kmh/ms, Kelvin->Celsius, compass direction)
  - WMO present-weather code descriptions
  - the sample-weighting function used to fit the Bayesian Ridge models
  - solar geometry (season-robust diurnal features)
  - XML namespace cleanup for DWD KML/KMZ parsing
  - ENGINEERED_FEATURES, the registry of dynamically-generated feature
    column names populated by CategorizedWindCorrectionPipeline._prepare_features

It has NO dependency on wingfoil_predictor.py, predict_html.py, or
CategorizedWindCorrectionPipeline.py -- everything else imports from here,
never the other way around. That direction matters: it's what makes it
possible to import CategorizedWindCorrectionPipeline (fit-side or
frozen-weights) from either predict_html.py or wingfoil_predictor.py without
circular imports, and it's what previously broke when
CategorizedWindCorrectionPipeline.py imported from predict_html.py.
"""
import json
import math
import os

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG_FILE = "config.json"

# Single canonical default config. Both the training/analysis entry point
# (wingfoil_predictor.py) and the live prediction entry point
# (predict_html.py) previously kept their own slightly different copies of
# this dict (predict_html.py's carried unused keys like init_valley_angle /
# bl_height_threshold_m left over from a hand-reimplementation that has
# since been removed -- see StandaloneWindPredictor's retirement). There is
# now exactly one copy, so a settings tweak here (or in config.json, which
# always wins) applies identically to both training and live prediction.
DEFAULT_CONFIG = {
    "settings": {
        "timezone": "Europe/Zurich",
        "wind_threshold_knots": 10.0,
        "foil_confirmed_threshold_knots": 10.0,
        "operational_window_utc": [11, 17],
        "rain_prob_confirm_threshold": 50.0,
    },
    "locations": {
        "davos": {"lat": 46.8041, "lon": 9.8372, "station_id": "06784"}
    },
    "category_feature_sets": {},
    "category_fx1_feature_sets": {},
}


def load_config(verbose=False):
    """Loads DEFAULT_CONFIG merged with user config from CONFIG_FILE
    (config.json), if present.

    Merge is per-key within each dict section: a key present only in
    DEFAULT_CONFIG survives even when config.json also defines that
    section, because config.json's keys only overwrite matching keys,
    never the whole section. Pass verbose=True to print which
    category_feature_sets / category_fx1_feature_sets keys came from disk
    vs. code defaults.

    Both wingfoil_predictor.py (training/analysis) and predict_html.py
    (live prediction) call this same function, so config.json is the one
    on-disk source of truth for both -- previously predict_html.py used a
    hardcoded CONFIG dict and never actually read config.json, so any
    tuning done there (rain_prob_confirm_threshold, foehn_stations,
    category_feature_sets, ...) silently applied to training but not to
    live prediction.
    """
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    disk_keys_by_section = {}

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            for section, values in user_config.items():
                if section in config and isinstance(config[section], dict) and isinstance(values, dict):
                    config[section].update(values)
                    disk_keys_by_section[section] = sorted(values.keys())
                else:
                    config[section] = values
        except Exception as e:
            print(f"⚠️ Warning: Erreur lors de la lecture de {CONFIG_FILE} : {e}")
    elif verbose:
        print(f"ℹ️ {CONFIG_FILE} not found. Using DEFAULT_CONFIG fallback.")

    if verbose:
        print("\n" + "=" * 80)
        print("🔧 CONFIG SOURCE MAP (category origin: disk vs. code default)")
        print("=" * 80)
        for section in ("category_feature_sets", "category_fx1_feature_sets"):
            all_cats = sorted(config.get(section, {}).keys())
            from_disk = set(disk_keys_by_section.get(section, []))
            print(f"\n📁 {section}:")
            for cat in all_cats:
                origin = "config.json" if cat in from_disk else "DEFAULT_CONFIG (code)"
                print(f"   • {cat:<28} <- {origin}")
        print("=" * 80)

    return config


def load_raw_config():
    """Reads raw JSON content directly from disk (no defaults merged in),
    for diff comparison (see get_config_differences)."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: '{CONFIG_FILE}' contains invalid JSON.")
            return {}
    return {}


def get_config_differences(old_config, new_config):
    """Recursively computes structural and value differences between
    configurations. Used by wingfoil_predictor.py's
    prompt_and_save_config_diff to show what a recommended config.json
    update would change."""
    diff = {"added": {}, "removed": {}, "modified": {}}

    all_keys = set(old_config.keys()) | set(new_config.keys())

    for key in all_keys:
        if key not in old_config:
            diff["added"][key] = new_config[key]
        elif key not in new_config:
            diff["removed"][key] = old_config[key]
        elif old_config[key] != new_config[key]:
            diff["modified"][key] = {
                "old": old_config[key],
                "new": new_config[key],
            }

    return {k: v for k, v in diff.items() if v}


# Global config instance, loaded once at import time (non-verbose). Callers
# needing the verbose source-map printout (wingfoil_predictor.py's main())
# call load_config(verbose=True) explicitly instead of relying on this.
CONFIG = load_config()


# =====================================================================
# DISPLAY / REGIME COLORS
# =====================================================================

# Palette for regime background shading (plots + HTML dashboard). Keyed by
# the exact classification strings CategorizedWindCorrectionPipeline
# produces (see get_combination_label) -- "NordFoehn", not "Foehn".
REGIME_COLORS = {
    "Sunny": "#fff9db",
    "PartlyCloudy": "#f1f3f5",
    "Cloudy": "#e9ecef",
    "NordFoehn": "#ffe3e3",
    "NordFoehn + Sunny": "#ffe8cc",
    "NordFoehn + PartlyCloudy": "#ffdeeb",
}
DEFAULT_REGIME_COLOR = "#ffffff"


# =====================================================================
# WMO PRESENT-WEATHER CODES
# =====================================================================

# WMO 4677 present-weather codes (as used by Open-Meteo / DWD MOSMIX), short
# labels for compact display. Covers 0-99; codes outside this range (or
# NaN) are handled by describe_weather_code() below.
WMO_WEATHER_CODES = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Dense drizzle",
    56: "Light frz drizzle", 57: "Dense frz drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Light frz rain", 67: "Frz rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Light snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "T-storm + hail", 99: "T-storm + heavy hail",
}


def describe_weather_code(w_code):
    """Returns a short text label for a WMO present-weather code, or 'N/A'
    if the code is missing/unrecognized."""
    if w_code is None or (isinstance(w_code, float) and pd.isna(w_code)):
        return "N/A"
    return WMO_WEATHER_CODES.get(int(w_code), f"Code {int(w_code)}")


# =====================================================================
# UNIT CONVERSIONS
# =====================================================================

def convert_ms_to_knots(ms):
    return ms * 1.94384 if ms is not None else None


def convert_kmh_to_knots(kmh):
    return kmh / 1.852 if kmh is not None else None


def kelvin_to_celsius(k):
    return k - 273.15 if k is not None else None


def degrees_to_cardinal(deg):
    """Converts wind direction degrees to a 16-point compass direction
    string, e.g. 'NNE (25°)'."""
    if deg is None or pd.isna(deg):
        return "-"
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((deg + 11.25) / 22.5) % 16
    return f"{dirs[idx]} ({deg:.0f}°)"


# =====================================================================
# XML / KML HELPERS
# =====================================================================

def clean_namespaces(root):
    """Recursively strips all XML namespaces to simplify tag lookup when
    parsing DWD MOSMIX KML files."""
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        for key in list(elem.attrib.keys()):
            if '}' in key:
                new_key = key.split('}', 1)[1]
                elem.attrib[new_key] = elem.attrib.pop(key)
    return root


# =====================================================================
# SOLAR GEOMETRY (season-robust diurnal features)
# =====================================================================

def solar_geometry(dt, lat, lon):
    """NOAA-simplified solar noon / sunrise / day-length (UTC hours) for a
    given date at (lat, lon).

    Used to build the season-robust daylight_frac_elapsed feature: raw
    clock hour is a proxy for thermal-cycle phase that's only valid near
    the season it was fit on (day length at Davos compresses from ~15.7h
    at the June solstice to ~9.8h by late October, which shifts and
    compresses the whole heating cycle). Computing phase-of-day from
    actual solar geometry lets the feature generalize across the
    operational June-October window instead of being implicitly tied to
    whatever months are in calibration_db.csv.

    This is the single implementation used by both
    CategorizedWindCorrectionPipeline (fit-side, calibration_db.csv) and
    live prediction -- previously predict_html.py kept a second,
    hand-copied version of this formula that had to be manually kept
    byte-for-byte in sync.
    """
    n = dt.timetuple().tm_yday
    B = math.radians(360 / 365 * (n - 81))
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)  # minutes
    solar_noon = 12 - lon / 15 - eot / 60
    decl = math.radians(23.44) * math.sin(math.radians(360 / 365 * (n - 81)))
    lat_r = math.radians(lat)
    cos_ha = max(-1.0, min(1.0, -math.tan(lat_r) * math.tan(decl)))
    ha = math.degrees(math.acos(cos_ha))
    day_len = 2 * ha / 15
    sunrise = solar_noon - day_len / 2
    return solar_noon, sunrise, day_len


# =====================================================================
# SHARED WEIGHTING FUNCTION
# =====================================================================

def compute_wingfoil_weights(y_true, threshold, max_weight=3.0, k=1.5):
    """
    Computes sample weights prioritizing wind speeds near and above
    `threshold`. `threshold` is the center of the sigmoid (weight ~2.0
    there); pass `foil_confirmed_threshold_knots - 0.5` from config so the
    weighting curve tracks the operational go/no-go threshold instead of a
    stale hardcoded value.
    e.g. with threshold=9.5 (i.e. foil threshold 10.0):
    - Wind <= 8.0 kt  : Weight ~ 1.0
    - Wind == 9.5 kt  : Weight ~ 2.0
    - Wind >= 11.0 kt : Weight ~ 3.0

    `threshold` is a required argument (no silent hardcoded default) so
    every call site is forced to state explicitly where it's sourcing it
    from.
    """
    y_array = np.asarray(y_true)
    w = 1.0 + (max_weight - 1.0) / (1.0 + np.exp(-k * (y_array - threshold)))
    w[y_array < 5] = 0.2
    return w


# =====================================================================
# ENGINEERED FEATURE REGISTRY
# =====================================================================

# Populated by CategorizedWindCorrectionPipeline._prepare_features as it
# generates dynamically-named columns (om_bl_height_higher_than_1400, the
# rotated-angle wind-direction feature families, etc). Shared as a module
# global -- imported by wingfoil_predictor.py's run_database_analysis,
# which uses it (via features_to_analyze) to know the full candidate
# feature list without hardcoding the generated names a second time.
ENGINEERED_FEATURES = []
