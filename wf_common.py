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
import io
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

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
        # Pressure-gradient (mosmix_dp_foehn, ZH - LU convention) magnitude
        # thresholds used by CategorizedWindCorrectionPipeline._prepare_features
        # to tag is_nordfoehn / is_sudfoehn. Kept as two separate settings
        # (not one shared "foehn_threshold_hpa") because the two regimes are
        # physically distinct and foehn_threshold_diagnostic.py's dedicated
        # confidence analysis found they don't share an optimal cutoff:
        # Nordfoehn's speed-boost signal is clean and direction-coherent at
        # ~3.5 hPa, while Sudfoehn's speed-reduction signal -- weaker and,
        # per the diagnostic's direction-clustering check, not strongly
        # direction-coherent -- currently only clears the diagnostic's
        # statistical bar at ~1.5 hPa. See that script's module docstring
        # and its "SUDFOEHN THRESHOLD -- DEDICATED CONFIDENCE ANALYSIS"
        # section before changing either value; rerun it as
        # calibration_db.csv grows to see whether the picture firms up.
        "nordfoehn_threshold_hpa": 3.5,
        "sudfoehn_threshold_hpa": 1.5,
    },
    "locations": {
        "davos": {"lat": 46.8041, "lon": 9.8372, "station_id": "06784", "ms_station_abbr": "DAV"}
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
    w[y_array < 5] = 0.5
    w[y_array < 2] = 0.2
    return w


# =====================================================================
# METEOSWISS (MS) STATION -- DAV (DAVOS), SwissMetNet OPEN DATA
# =====================================================================
#
# MeteoSwiss's SwissMetNet Open Data API (data.geo.admin.ch) publishes
# real OBSERVED (not forecast) hourly station data, free, no API key.
# "MS" is this toolchain's abbreviation for MeteoSwiss, and "DAV" is
# MeteoSwiss's own three-letter code for the Davos station -- the same
# physical site as DSSC (see locations.davos in DEFAULT_CONFIG), so MS is
# a second, independent observation source for the same location rather
# than a different site like the Chur/Bad Ragaz stations explored for
# feature-engineering experiments elsewhere in this project's history.
#
# MS (MeteoSwiss DAV) is now the DEFAULT ground truth for feature
# selection, fitting, replay, and the normal run, via
# CategorizedWindCorrectionPipeline._select_ground_truth, which copies
# ms_* into the internal obs_* working columns by default. A CLI flag
# (--dssc in wingfoil_predictor.py) opts a run into having that same
# selection step use dssc_* columns as the ground truth for feature
# selection / Bayesian Ridge fitting / replay instead of ms_*, for
# comparing the two observation sources' effect on the model rather
# than blending them silently. Neither ms_* nor dssc_* should be read
# directly for feature selection/fitting anywhere else -- obs_* (as
# selected by _select_ground_truth) is the only name that should touch
# that code path. DSSC's dssc_* columns remain available for
# display/comparison regardless of which source is selected for fitting.
#
# MeteoSwiss explicitly recommends using their pre-aggregated HOURLY (h)
# files rather than re-aggregating the 10-minute series by hand: the
# hourly files can include corrections not present in the raw 10-minute
# data. This module therefore always fetches the "_h_" (hourly) file, not
# the "_t_" (10-minute) file.
#
# TIMESTAMP CONVENTION -- READ BEFORE TOUCHING THIS SECTION:
# MeteoSwiss's reference_timestamp for hourly files is the END of the
# hourly interval, UTC. E.g. "01.09.2026 14:00" is the hour that RAN FROM
# 13:00 TO 14:00 UTC. DSSC/calibration_db.csv's "hour" column, by
# contrast, is the START of the hour (see wingfoil_predictor.py's
# process_dssc_hourly, which buckets raw sub-hourly DSSC readings by the
# hour they fall within, i.e. "13:00" holds everything from 13:00:00 to
# 13:59:59). To align MS with DSSC/calibration_db.csv's existing
# convention, every MS timestamp is shifted back by exactly one hour
# (interval-end -> interval-start) at parse time in fetch_ms_hourly_data
# below. Get this wrong and every MS reading will be silently offset by
# one hour relative to DSSC and the MOSMIX/OpenMeteo forecast columns it's
# being compared against.
#
# TIMEZONE -- SECOND, SEPARATE GOTCHA:
# The interval-start shift above only fixes the interval convention; it
# does NOT touch timezone. MeteoSwiss timestamps are UTC (confirmed by
# MeteoSwiss's own Open Data docs), so after the shift the index is still
# UTC. But every consumer of fetch_ms_hourly_data's output -- the "hour"
# column of calibration_db.csv, DSSC's dssc_* (built from a station clock
# treated as local time in process_dssc_hourly), and MOSMIX/OpenMeteo's
# df_meteo index (explicitly tz_convert'd to Europe/Zurich in
# fetch_mosmix_davos_data) -- key their hours in Europe/Zurich LOCAL time,
# not UTC. Switzerland is UTC+1 (CET) or UTC+2 (CEST), so leaving the MS
# index in UTC silently shifts every MS observation by 1-2 hours relative
# to DSSC/MOSMIX when looked up by "HH:00" string, causing MS to appear
# mostly missing (wrong hours don't match) and the few hours that do
# coincide to hold the wrong hour's reading. fetch_ms_hourly_data
# therefore converts the shifted UTC index to Europe/Zurich local time
# (then drops the tzinfo, since every other index in this toolchain is
# tz-naive local time) before returning.
MS_STAC_BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
MS_DEFAULT_STATION_ABBR = "DAV"  # Davos -- same site as DSSC/locations.davos

# Column names as confirmed against a live fetch of the MeteoSwiss hourly
# "recent" file for a Swiss automatic weather station -- stable across
# stations sharing the same instrumentation generation, but
# fetch_ms_hourly_data checks for their presence rather than assuming
# blindly, since a station could in principle expose a different subset.
MS_COL_WIND_SPEED = "fu3010h0"   # hourly mean wind speed, km/h
MS_COL_WIND_GUST = "fu3010h1"    # hourly max gust, km/h
MS_COL_WIND_DIR = "dkl010h0"     # hourly mean wind direction, degrees


def fetch_ms_hourly_data(station_abbr=MS_DEFAULT_STATION_ABBR, historical_years=None):
    """Fetches MeteoSwiss SwissMetNet hourly station data (real
    observations, not forecast) for the given station abbreviation.

    Always uses the pre-aggregated hourly ("_h_") file per MeteoSwiss's own
    recommendation (see module-level comment above) -- never re-aggregates
    from the 10-minute series.

    Returns a DataFrame indexed by INTERVAL-START UTC timestamp (already
    shifted back one hour from MeteoSwiss's interval-end convention, see
    module docstring), with columns:
        ms_speed_kt, ms_gust_kt  -- converted from km/h to knots
        ms_dir_deg               -- degrees, unchanged
    or None if the fetch fails entirely.

    `historical_years`: optional list of year-range strings (e.g.
    ["2020-2023"]) to additionally fetch from the "_h_historical_" archive
    files, for backfilling dates older than the "recent" file's rolling
    window. The "recent" file is always fetched and included regardless of
    this argument.
    """
    abbr_lower = station_abbr.lower()
    frames = []

    recent_url = f"{MS_STAC_BASE_URL}/{abbr_lower}/ogd-smn_{abbr_lower}_h_recent.csv"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        resp = requests.get(recent_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text), sep=";"))
    except Exception as e:
        print(f"⚠️ Warning: could not fetch MS (MeteoSwiss) recent data for "
              f"station '{station_abbr}': {e}")

    if historical_years:
        for year_range in historical_years:
            hist_url = f"{MS_STAC_BASE_URL}/{abbr_lower}/ogd-smn_{abbr_lower}_h_historical_{year_range}.csv"
            try:
                resp = requests.get(hist_url, headers=HEADERS, timeout=10)
                resp.raise_for_status()
                frames.append(pd.read_csv(io.StringIO(resp.text), sep=";"))
            except Exception as e:
                print(f"⚠️ Warning: could not fetch MS historical data "
                      f"({year_range}) for station '{station_abbr}': {e}")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]

    if "reference_timestamp" not in df.columns:
        print(f"⚠️ Warning: MS data for '{station_abbr}' has no "
              f"'reference_timestamp' column -- got {list(df.columns)}")
        return None

    # MeteoSwiss hourly timestamps are typically "DD.MM.YYYY HH:MM" (day-
    # first). dayfirst=True is REQUIRED here -- without it pandas silently
    # misparses e.g. "01.09.2026" (1 September) as if month=01, day=09 for
    # any day <=12, corrupting exactly the ambiguous dates.
    df["_dt_end"] = pd.to_datetime(df["reference_timestamp"], dayfirst=True, errors="coerce")
    if df["_dt_end"].isna().mean() > 0.5:
        # fallback for the compact "YYYYMMDDHHMM" format some MeteoSwiss
        # exports use instead
        df["_dt_end"] = pd.to_datetime(df["reference_timestamp"], format="%Y%m%d%H%M", errors="coerce")

    df = df.dropna(subset=["_dt_end"])

    # THE CRITICAL SHIFT: interval-end -> interval-start, see module
    # docstring. This is what makes MS timestamps directly comparable to
    # calibration_db.csv's "hour" column and DSSC's dssc_* columns.
    dt_start_utc = df["_dt_end"] - pd.Timedelta(hours=1)

    # SECOND CRITICAL STEP -- UTC -> Europe/Zurich local, see module
    # docstring's "TIMEZONE" note. Without this, MS hours are compared
    # against DSSC/MOSMIX's local-time hour keys while still in UTC,
    # silently offsetting every MS reading by 1h (CET) or 2h (CEST).
    df["datetime"] = (
        dt_start_utc.dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Zurich")
        .dt.tz_localize(None)
    )

    missing = [c for c in (MS_COL_WIND_SPEED, MS_COL_WIND_GUST, MS_COL_WIND_DIR) if c not in df.columns]
    if missing:
        print(f"⚠️ Warning: MS data for '{station_abbr}' is missing expected "
              f"column(s) {missing} -- got {list(df.columns)}. Returning "
              f"whatever subset is available.")

    out = pd.DataFrame(index=df["datetime"])
    if MS_COL_WIND_SPEED in df.columns:
        out["ms_speed_kt"] = convert_kmh_to_knots(df[MS_COL_WIND_SPEED].values)
    if MS_COL_WIND_GUST in df.columns:
        out["ms_gust_kt"] = convert_kmh_to_knots(df[MS_COL_WIND_GUST].values)
    if MS_COL_WIND_DIR in df.columns:
        out["ms_dir_deg"] = df[MS_COL_WIND_DIR].values

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


MS_COL_WIND_SPEED_10MIN = "fu3010z0"  # ten-minute mean wind speed, km/h
MS_COL_WIND_GUST_10MIN = "fu3010z1"   # ten-minute max gust, km/h
MS_COL_WIND_DIR_10MIN = "dkl010z0"    # ten-minute mean wind direction, degrees


def fetch_ms_now_data(station_abbr=MS_DEFAULT_STATION_ABBR):
    """Fetches MeteoSwiss SwissMetNet ten-minute ("_t_now_") station data --
    the same real-observation source as fetch_ms_hourly_data, but at the
    station's native 10-minute resolution and updated since midnight
    local time (today only), rather than the pre-aggregated hourly file.

    This exists ONLY to support "today so far": the hourly "_h_recent_"
    file fetch_ms_hourly_data reads lags behind by up to an hour, so for
    the CURRENT day's not-yet-complete hour there is no hourly reading
    yet. The 10-minute "now" file lets a live run compute a same-hour
    running average and plot the finer-grained curve, without touching
    how past days or replay work (they still use fetch_ms_hourly_data
    exclusively).

    Returns a DataFrame indexed by INTERVAL-START UTC-shifted, then
    Europe/Zurich-local timestamp -- same two-step convention as
    fetch_ms_hourly_data (interval-end -> interval-start, then UTC ->
    Europe/Zurich local) -- with columns:
        ms_speed_kt, ms_gust_kt  -- converted from km/h to knots
        ms_dir_deg               -- degrees, unchanged
    or None if the fetch fails entirely.
    """
    abbr_lower = station_abbr.lower()
    now_url = f"{MS_STAC_BASE_URL}/{abbr_lower}/ogd-smn_{abbr_lower}_t_now.csv"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        resp = requests.get(now_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep=";")
    except Exception as e:
        print(f"⚠️ Warning: could not fetch MS (MeteoSwiss) 10-minute 'now' "
              f"data for station '{station_abbr}': {e}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]

    if "reference_timestamp" not in df.columns:
        print(f"⚠️ Warning: MS 'now' data for '{station_abbr}' has no "
              f"'reference_timestamp' column -- got {list(df.columns)}")
        return None

    # Same day-first parsing caveat as fetch_ms_hourly_data.
    df["_dt_end"] = pd.to_datetime(df["reference_timestamp"], dayfirst=True, errors="coerce")
    if df["_dt_end"].isna().mean() > 0.5:
        df["_dt_end"] = pd.to_datetime(df["reference_timestamp"], format="%Y%m%d%H%M", errors="coerce")

    df = df.dropna(subset=["_dt_end"])

    # Same interval-end -> interval-start shift as fetch_ms_hourly_data,
    # scaled to the 10-minute granularity here instead of 1 hour -- see
    # that function's docstring/module comment for why this shift and
    # the UTC -> Europe/Zurich conversion below are both required.
    dt_start_utc = df["_dt_end"] - pd.Timedelta(minutes=10)
    df["datetime"] = (
        dt_start_utc.dt.tz_localize("UTC")
        .dt.tz_convert("Europe/Zurich")
        .dt.tz_localize(None)
    )

    missing = [c for c in (MS_COL_WIND_SPEED_10MIN, MS_COL_WIND_GUST_10MIN, MS_COL_WIND_DIR_10MIN) if c not in df.columns]
    if missing:
        print(f"⚠️ Warning: MS 'now' data for '{station_abbr}' is missing "
              f"expected column(s) {missing} -- got {list(df.columns)}. "
              f"Returning whatever subset is available.")

    out = pd.DataFrame(index=df["datetime"])
    if MS_COL_WIND_SPEED_10MIN in df.columns:
        out["ms_speed_kt"] = convert_kmh_to_knots(pd.to_numeric(df[MS_COL_WIND_SPEED_10MIN], errors="coerce").values)
    if MS_COL_WIND_GUST_10MIN in df.columns:
        out["ms_gust_kt"] = convert_kmh_to_knots(pd.to_numeric(df[MS_COL_WIND_GUST_10MIN], errors="coerce").values)
    if MS_COL_WIND_DIR_10MIN in df.columns:
        out["ms_dir_deg"] = pd.to_numeric(df[MS_COL_WIND_DIR_10MIN], errors="coerce").values

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def compute_ms_hourly_so_far(ms_now_df, target_date):
    """Builds the SAME {"HH:00": {"speed":..., "gust":..., "dir":...}}
    shape as get_ms_hourly_for_date, but from the 10-minute "now" series
    (fetch_ms_now_data) instead of the pre-aggregated hourly file --
    i.e. a running average over whatever 10-minute samples have arrived
    so far for each hour, for the CURRENT, not-yet-complete hour.

    Each 10-minute sample already carries the interval-start, Europe/
    Zurich-local timestamp (see fetch_ms_now_data), so a sample at, say,
    13:40 belongs to the "13:00" hourly bucket exactly like
    fetch_ms_hourly_data's own hour-start convention -- "the average is
    from the data from the coming hour" in the sense that a 10-minute
    reading whose OWN raw reference_timestamp (interval-end, before the
    shift) falls at HH:50 is the last sample of hour HH, and a reading
    at (HH+1):00 is the first sample counted towards the NEXT hour's
    average, matching fetch_ms_hourly_data's bucketing exactly so the
    two are directly comparable once the hourly file catches up.

    Wind direction is circular-averaged (same approach as
    process_dssc_hourly), speed/gust are arithmetic means of the
    10-minute mean-speed / peak-gust samples collected so far -- an
    approximation of the true hourly aggregate (which MeteoSwiss
    computes from the underlying 1-second data, not from the six
    10-minute values), acceptable here since this is only ever used as a
    live, provisional stand-in for the current, still-incomplete hour.

    target_date: "YYYY-MM-DD" string.
    Returns {} if ms_now_df is None or has no rows for that date.
    """
    if ms_now_df is None or ms_now_df.empty:
        return {}

    day_mask = ms_now_df.index.strftime("%Y-%m-%d") == target_date
    day_df = ms_now_df[day_mask]
    if day_df.empty:
        return {}

    result = {}
    for hour_str, group in day_df.groupby(day_df.index.strftime("%H:00")):
        speeds = group["ms_speed_kt"].dropna() if "ms_speed_kt" in group else pd.Series(dtype=float)
        gusts = group["ms_gust_kt"].dropna() if "ms_gust_kt" in group else pd.Series(dtype=float)
        dirs = group["ms_dir_deg"].dropna() if "ms_dir_deg" in group else pd.Series(dtype=float)

        avg_dir = None
        if not dirs.empty:
            sin_sum = np.sin(np.radians(dirs)).sum()
            cos_sum = np.cos(np.radians(dirs)).sum()
            R = np.hypot(sin_sum, cos_sum)
            if R > 1e-5:
                avg_dir = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360)

        result[hour_str] = {
            "speed": float(speeds.mean()) if not speeds.empty else None,
            "gust": float(gusts.max()) if not gusts.empty else None,
            "dir": avg_dir,
        }
    return result


def get_ms_hourly_for_date(ms_df, target_date):
    """Slices a fetch_ms_hourly_data() result down to a single date and
    reshapes it into the same {"HH:00": {"speed":..., "gust":...,
    "dir":...}} dict shape process_dssc_hourly() produces, so callers
    (analyze_day, analyze_day_replay, generate_day_graph, ...) can treat
    ms_hourly exactly like dssc_hourly -- same key format, same per-hour
    dict shape, same None-for-missing-hour semantics.

    target_date: "YYYY-MM-DD" string.
    Returns {} if ms_df is None or has no rows for that date.
    """
    if ms_df is None or ms_df.empty:
        return {}

    day_mask = ms_df.index.strftime("%Y-%m-%d") == target_date
    day_df = ms_df[day_mask]
    if day_df.empty:
        return {}

    result = {}
    for ts, row in day_df.iterrows():
        hour_str = ts.strftime("%H:00")
        result[hour_str] = {
            "speed": row.get("ms_speed_kt") if pd.notna(row.get("ms_speed_kt")) else None,
            "gust": row.get("ms_gust_kt") if pd.notna(row.get("ms_gust_kt")) else None,
            "dir": row.get("ms_dir_deg") if pd.notna(row.get("ms_dir_deg")) else None,
        }
    return result


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
