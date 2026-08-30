# =====================================================================
# CATEGORIZED WIND CORRECTION PIPELINE (BAYESIAN RIDGE)
# =====================================================================
import math
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge

# Common utilities, configuration defaults, and the engineered-feature
# registry live in wf_common.py -- the one shared foundation module with no
# dependency on any of the three entry-point scripts. This module used to
# import from predict_html, which was backwards (a shared module should
# never depend on one of its two consumers) and meant this file couldn't be
# imported at all without pulling in predict_html's argparse/matplotlib/
# HTML-generation code. See wf_common.py's module docstring.
from wf_common import (
    DEFAULT_CONFIG,
    ENGINEERED_FEATURES,
    load_config,
    compute_wingfoil_weights,
)


class CategorizedWindCorrectionPipeline:
    """
    Production-grade Wind Correction Pipeline combining:
    1. Category-specific Bayesian Ridge Regressors (for mean + variance)
    2. Multiplicative Damping Factor for Rain / Convective events
    3. Category Bias Offsets & Global Fallbacks for low-sample regimes
    
    Trained strictly on historical operational hours (11:00 - 17:00).
    """
    def __init__(self, db_filepath="calibration_db.csv", min_samples_for_ridge=10, verbose_summary=False):
        self.db_filepath = db_filepath
        self.min_samples_for_ridge = min_samples_for_ridge
        self.verbose_summary = verbose_summary
        
        config = load_config()
        self.op_window = config["settings"].get("operational_window_utc", [11, 17])
        
        self.foil_threshold = config["settings"].get("foil_confirmed_threshold_knots", 10.0)
        self.weight_center = self.foil_threshold - 0.5  

        # Foehn pressure-gradient magnitude thresholds (mosmix_dp_foehn,
        # ZH - LU convention: positive => Nordfoehn, negative => Sudfoehn).
        # Deliberately two independent settings, not a shared magnitude --
        # see DEFAULT_CONFIG's comment in wf_common.py and
        # foehn_threshold_diagnostic.py's dedicated Sudfoehn confidence
        # analysis for why the two sides don't share an optimal cutoff.
        self.nordfoehn_threshold_hpa = config["settings"].get("nordfoehn_threshold_hpa", 3.5)
        self.sudfoehn_threshold_hpa = config["settings"].get("sudfoehn_threshold_hpa", 1.5)

        # Shared confidence threshold for rain-bearing WMO codes -- see
        # DEFAULT_CONFIG comment and is_rain_damping_code / classify_precip_type.
        # Read once here so classification and damping consult the exact same
        # value instead of two hardcoded literals that could drift apart.
        self.rain_prob_confirm_threshold = config["settings"].get(
            "rain_prob_confirm_threshold", 50.0
        )

        # Station coordinates, used to compute season-robust solar-relative
        # diurnal features (see solar_geometry / _add_diurnal_features).
        # Falls back to the Davos config entry since that's currently the
        # only location this pipeline is fit/applied to.
        davos_loc = config.get("locations", {}).get("davos", {})
        self.station_lat = davos_loc.get("lat", 46.8041)
        self.station_lon = davos_loc.get("lon", 9.8372)

        # 2. Category-specific predictor feature registries from config,
        # falling back to DEFAULT_CONFIG (single source of truth, see top of
        # file) rather than a second hardcoded copy that could silently
        # diverge from it.
        self.category_feature_sets = config.get(
            "category_feature_sets", DEFAULT_CONFIG["category_feature_sets"]
        )
        self.category_fx1_feature_sets = config.get(
            "category_fx1_feature_sets", DEFAULT_CONFIG["category_fx1_feature_sets"]
        )
        
        # Internal registries
        self.bayesian_models = {}
        self.category_mean_offsets = {}
        self.category_std_offsets = {}
        self.global_fallback_bias = 0.0
        self.global_std_bias = 0.0
        
        self.bayesian_fx1_models = {}
        self.category_fx1_mean_offsets = {}
        self.category_fx1_std_offsets = {}
        self.global_fx1_fallback_bias = 0.0
        self.global_fx1_std_bias = 0.0
        
        # Fit models on initialization if database is available
        if self.db_filepath:
            self.fit_from_database()

    @classmethod
    def from_exported_weights(cls, weights, min_samples_for_ridge=10):
        """Reconstructs a pipeline instance from a model_weights.json-style
        dict (see export_weights_dict/export_weights_to_json) instead of
        fitting from calibration_db.csv.

        This is what makes predict_html.py's live prediction path and
        wingfoil_predictor.py's training/analysis path run the *same*
        process()/_prepare_features()/classify_precip_type()/
        get_delta_breakdown() code. predict_html.py previously carried its
        own StandaloneWindPredictor class that reimplemented all of that by
        hand against the exported JSON -- correct at the time it was
        written, but with no mechanism to stay correct as the fit-side
        pipeline evolved (it had already drifted: e.g. it used a single
        om_bl_height threshold and a valley-angle feature family that
        aren't part of this pipeline's actual feature set). Loading the
        exported scaler/coef/intercept/sigma/alpha directly into real
        sklearn objects means there is exactly one implementation of the
        correction math, exercised identically whether the pipeline was
        just fit from calibration_db.csv or reconstructed from a JSON
        export produced by an earlier fit.

        Rebuilds StandardScaler + BayesianRidge objects with the exact
        fitted parameters (mean_/scale_ on the scaler; coef_/intercept_/
        alpha_/sigma_ on the ridge) rather than recomputing predictions
        from raw numbers, so BayesianRidge.predict(..., return_std=True)
        -- including the predictive std formula -- runs unmodified.
        """
        self = cls.__new__(cls)  # bypass __init__ / fit_from_database

        self.db_filepath = None
        self.min_samples_for_ridge = min_samples_for_ridge
        self.verbose_summary = False

        config = load_config()
        self.op_window = config["settings"].get("operational_window_utc", [11, 17])
        self.foil_threshold = config["settings"].get("foil_confirmed_threshold_knots", 10.0)
        self.weight_center = self.foil_threshold - 0.5
        self.rain_prob_confirm_threshold = config["settings"].get(
            "rain_prob_confirm_threshold", 50.0
        )
        # Foehn pressure-gradient magnitude thresholds -- see the matching
        # comment in __init__ / wf_common.py's DEFAULT_CONFIG. Must be set
        # here too since from_exported_weights bypasses __init__ entirely
        # (cls.__new__(cls)); without it, live prediction's _prepare_features
        # call would hit an AttributeError the moment it tried to tag
        # is_nordfoehn / is_sudfoehn.
        self.nordfoehn_threshold_hpa = config["settings"].get("nordfoehn_threshold_hpa", 3.5)
        self.sudfoehn_threshold_hpa = config["settings"].get("sudfoehn_threshold_hpa", 1.5)
        davos_loc = config.get("locations", {}).get("davos", {})
        self.station_lat = davos_loc.get("lat", 46.8041)
        self.station_lon = davos_loc.get("lon", 9.8372)
        self.category_feature_sets = dict(config.get(
            "category_feature_sets", DEFAULT_CONFIG["category_feature_sets"]
        ))
        self.category_fx1_feature_sets = dict(config.get(
            "category_fx1_feature_sets", DEFAULT_CONFIG["category_fx1_feature_sets"]
        ))

        self.global_fallback_bias = weights.get("global_fallback_bias", 0.0)
        self.global_std_bias = weights.get("global_std_bias", 0.0)
        self.global_fx1_fallback_bias = weights.get("global_fx1_fallback_bias", 0.0)
        self.global_fx1_std_bias = weights.get("global_fx1_std_bias", 0.0)
        self.category_mean_offsets = dict(weights.get("category_mean_offsets", {}))
        self.category_std_offsets = dict(weights.get("category_std_offsets", {}))
        self.category_fx1_mean_offsets = dict(weights.get("category_fx1_mean_offsets", {}))
        self.category_fx1_std_offsets = dict(weights.get("category_fx1_std_offsets", {}))

        # The scaler/ridge rebuilt below for each category were fit on the
        # exact column names/order stored in the exported weights' own
        # "features" key -- that, not config.json, is the ground truth for
        # what a *fitted* model expects. config.json's category_feature_sets
        # is allowed to drift from model_weights.json between deploys (e.g.
        # a stale feature name left over from an earlier feature-engineering
        # version, or a re-export that changed the selected features for a
        # category); when it does, process() would otherwise build the
        # prediction-time DataFrame using config.json's (wrong) column
        # names, and sklearn's fitted-feature-name check on the frozen
        # scaler raises ValueError. Overriding per-category here from the
        # weights themselves keeps process() structurally unable to ask a
        # rebuilt model for columns it wasn't fit on. config.json's entries
        # remain the fallback for any category with no fitted model.
        for cat, m in weights.get("bayesian_models", {}).items():
            if m.get("features"):
                self.category_feature_sets[cat] = list(m["features"])
        for cat, m in weights.get("bayesian_fx1_models", {}).items():
            if m.get("features"):
                self.category_fx1_feature_sets[cat] = list(m["features"])

        self.bayesian_models = self._rebuild_pipelines(weights.get("bayesian_models", {}))
        self.bayesian_fx1_models = self._rebuild_pipelines(weights.get("bayesian_fx1_models", {}))

        return self

    @staticmethod
    def _rebuild_pipelines(exported_model_dict):
        """Reconstructs {category: sklearn Pipeline} from the serialized
        form written by export_weights_dict's serialize_pipeline. Sets
        fitted attributes directly (mean_/scale_/coef_/intercept_/alpha_/
        sigma_) rather than re-fitting, since the whole point is to
        reproduce a previously-fitted model exactly."""
        rebuilt = {}
        for cat, m in exported_model_dict.items():
            features = m.get("features", [])

            scaler = StandardScaler()
            scaler.mean_ = np.array(m["scaler_mean"])
            scaler.scale_ = np.array(m["scaler_scale"])
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            if features:
                # Matches the feature-name dtype sklearn stores when a
                # pipeline is fit on a DataFrame (as process() always
                # passes), which avoids a spurious "fitted without feature
                # names" warning when predicting on a DataFrame here.
                scaler.feature_names_in_ = np.array(features, dtype=object)

            ridge = BayesianRidge(compute_score=True)
            ridge.coef_ = np.array(m["coef"])
            ridge.intercept_ = float(m["intercept"])
            ridge.alpha_ = float(m["alpha"])
            ridge.sigma_ = np.array(m["sigma"])
            ridge.n_features_in_ = len(ridge.coef_)
            # Newer sklearn versions (see BayesianRidge.predict's
            # return_std=True branch) center X on self.X_offset_ before
            # applying sigma_ to get the predictive std -- an attribute
            # that's only ever set inside .fit() and was never part of
            # the exported weights dict, so a reconstructed-from-JSON
            # ridge raised AttributeError the first time predict() was
            # called with return_std=True. Since this ridge is always fed
            # already-standardized input (the StandardScaler step runs
            # first in the pipeline, and it was fit on that same
            # zero-mean data), the correct reconstruction is a zero
            # offset -- not an arbitrary placeholder.
            ridge.X_offset_ = np.zeros_like(ridge.coef_)

            pipe = make_pipeline(scaler, ridge)
            rebuilt[cat] = pipe
        return rebuilt

    @staticmethod
    def is_rain_damping_code(w_code):
        """WMO present-weather codes treated as rain-bearing for the purpose
        of apply_rain_multiplicative_factor. Single source of truth for the
        "is this a rain code" check -- classify_precip_type and the damping
        factor both call this instead of each keeping their own copy of the
        (50-69)/(80-82) range, which is how they drifted apart before."""
        if pd.isna(w_code):
            return False
        w = int(w_code)
        return (50 <= w <= 69) or (80 <= w <= 82)

    def classify_precip_type(self, row):
        """Labels the precipitation regime for a row.

        "Rain" is only returned when the WMO code is rain-bearing AND
        om_prec_prob clears self.rain_prob_confirm_threshold -- the exact
        same two conditions apply_rain_multiplicative_factor requires to
        actually damp the forecast. This keeps classification and the
        physical correction consistent by construction: a row can no longer
        be tagged "Precip(Rain)" while the damping factor silently leaves it
        undamped (or vice versa).

        Rain-coded rows that don't clear the probability bar get their own
        "Rain_LowConf" label instead of being silently folded into "Rain" or
        dropped back to "No_Precip" -- the rain code is real signal even at
        low ensemble confidence, and get_combination_label /
        fit_from_database use this label to fit a dedicated (small-sample)
        correction for exactly the hours that escape damping, rather than
        letting them fall through resolve_category to a dry-weather model
        with no precip signal at all.
        """
        if not row.get("is_precip", False):
            return "No_Precip"

        w_code = row.get("om_w_codes")
        if pd.isna(w_code):
            return "Precip_Unknown"

        w_code = int(w_code)
        prec_prob = row.get("om_prec_prob", np.nan)

        if self.is_rain_damping_code(w_code):
            if pd.notna(prec_prob) and prec_prob > self.rain_prob_confirm_threshold:
                return "Rain"
            return "Rain_LowConf"
        elif (70 <= w_code <= 79) or (85 <= w_code <= 86):
            return "Snow"
        elif 90 <= w_code <= 99:
            return "Thunderstorm"
        else:
            return "Other_Precip"
            
    @classmethod
    def get_combination_label(cls, row):
        """Build categorical tags based on weather state."""
        tags = []
        if row.get("is_nordfoehn", False):
            tags.append("NordFoehn")
        elif row.get("is_sudfoehn", False):
            tags.append("Sudfoehn")
        if row.get("is_sunny", False):
            tags.append("Sunny")
        elif row.get("is_partly_cloudy", False):
            tags.append("PartlyCloudy")
        elif row.get("is_cloudy", False):
            tags.append("Cloudy")
        if row.get("is_precip", False):
            tags.append(f"Precip({row.get('precip_type', 'No_Precip')})")
        return " + ".join(tags) if tags else "Unclassified"

    @staticmethod
    def solar_geometry(dt, lat, lon):
        """NOAA-simplified solar noon / sunrise / day-length (UTC hours) for
        a given date at (lat, lon). Used to build season-robust diurnal
        features: raw clock hour is a proxy for thermal-cycle phase that's
        only valid near the season it was fit on (day length at Davos
        compresses from ~15.7h at the June solstice to ~9.8h by late
        October, which shifts and compresses the whole heating cycle).
        Computing phase-of-day from actual solar geometry lets the feature
        generalize across the operational June-October window instead of
        being implicitly tied to whatever months are in calibration_db.csv.
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

    def _add_diurnal_features(self, df):
        """Adds hour_int (raw clock hour, UTC) and daylight_frac_elapsed
        (season-robust: position in the daylight period, 0=sunrise,
        1=sunset, computed from date + station lat/lon) as candidate
        features for feature-set selection. Both are exposed side by side
        -- config-driven feature-set selection (category_feature_sets /
        run_database_analysis's recommend_features_robust) picks whichever
        actually scores better rather than this function deciding for it.

        Derives date/hour from either 'date'/'hour' columns (calibration DB
        path) or a DatetimeIndex (live MOSMIX prediction path), so the
        features are available identically in fit, replay/analysis, and
        live prediction.
        """
        if "hour_int" in df.columns:
            hour_int = df["hour_int"]
        elif "hour" in df.columns:
            hour_int = df["hour"].apply(lambda x: int(str(x).split(":")[0]) if pd.notna(x) else np.nan)
        elif isinstance(df.index, pd.DatetimeIndex):
            hour_int = pd.Series(df.index.hour, index=df.index)
        else:
            hour_int = pd.Series(np.nan, index=df.index)

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = pd.Series(df.index.normalize(), index=df.index)
        else:
            dates = pd.Series(pd.NaT, index=df.index)

        df["hour_int"] = hour_int.values if hasattr(hour_int, "values") else hour_int

        lat = self.station_lat
        lon = self.station_lon

        def _daylight_frac(row_hour, row_date):
            if pd.isna(row_hour) or pd.isna(row_date):
                return np.nan
            ts = pd.Timestamp(row_date)
            if pd.isna(ts):
                return np.nan
            _, sunrise, day_len = self.solar_geometry(ts.to_pydatetime(), lat, lon)
            if day_len <= 0:
                return np.nan
            return (row_hour - sunrise) / day_len

        df["daylight_frac_elapsed"] = [
            _daylight_frac(h, d) for h, d in zip(df["hour_int"], dates)
        ]

        return df

    def _prepare_features(self, df_data):
        df = df_data.copy()

        # Basic atmospheric regime thresholds.
        # mosmix_dp_foehn follows the ZH - LU convention from foehn_gradient.py:
        # positive => Nordfoehn (the regime that actually brings good foiling
        # conditions in Davos), negative => Sudfoehn (opposite synoptic setup,
        # kept as its own tag rather than merged with "Foehn" so it doesn't
        # dilute the Foehn categories' learned bias correction with a
        # physically different regime).
        #
        # Nordfoehn and Sudfoehn use SEPARATE magnitude thresholds, not a
        # shared |dp_foehn| cutoff -- see self.nordfoehn_threshold_hpa /
        # self.sudfoehn_threshold_hpa (config-driven, see wf_common.py's
        # DEFAULT_CONFIG comment) and foehn_threshold_diagnostic.py's
        # dedicated Sudfoehn confidence analysis. That analysis's current
        # verdict on the Sudfoehn side is SUGGESTIVE, not CONFIDENT: the
        # 1.5 hPa cutoff clears the statistical bar (Welch's t-test +
        # bootstrap CI both exclude zero) but fails the physical
        # cross-check (obs_dir_deg does not clearly cluster there,
        # clustering r ~0.2 vs Nordfoehn's ~0.8) -- rerun that script as
        # calibration_db.csv grows before trusting this value further.
        df["is_nordfoehn"] = df["mosmix_dp_foehn"] > self.nordfoehn_threshold_hpa
        df["is_sudfoehn"] = df["mosmix_dp_foehn"] < -self.sudfoehn_threshold_hpa
        df["is_sunny"] = df["mosmix_cloud_pct"] < 33.0
        df["is_partly_cloudy"] = (df["mosmix_cloud_pct"] >= 33.0) & (
            df["mosmix_cloud_pct"] <= 66.0
        )
        df["is_cloudy"] = df["mosmix_cloud_pct"] > 66.0
        # Precip regime tag now keys off the forecast weather code (om_w_codes)
        # rather than om_prec_prob. Probability alone was fragmenting the
        # training data: high-probability/no-rain-code hours (e.g. overcast,
        # code 3, 88% prob) were getting tagged into sparse composite regimes
        # like "PartlyCloudy + Precip(Other_Precip)" that rarely have enough
        # samples for their own model. Codes 50-99 cover drizzle/rain/snow/
        # thunderstorm -- the same ranges classify_precip_type already uses,
        # so is_precip and precip_type are now consistent with each other.
        df["is_precip"] = df["om_w_codes"].between(50, 99)

        df["precip_type"] = df.apply(self.classify_precip_type, axis=1)
        df["classification"] = df.apply(self.get_combination_label, axis=1)

        # Binary boundary-layer-height regime flag (see comment in __init__).
        # Kept as its own column (not a replacement for raw om_bl_height) so
        # both the step feature and the continuous value stay available as
        # separate candidates for feature selection.
        heights = [1400, 1600, 1800]
        for height in heights:
            suffix = "" if height == 0 else f"_higher_than_{height}"
            name = f"om_bl_height{suffix}"
            df[f"{name}"] = (df["om_bl_height"] >= height).astype(float)
            if name not in ENGINEERED_FEATURES:
                ENGINEERED_FEATURES.append(name)
        
        # Generate a list of features from om_wind_speed_10m and om_wind_direction_10m for different angles 
        angles = [] # V20: currently swicthed off, it doesn't seem to bring too much
        for angle in angles:
            # Handle angle suffix formatting (e.g., angle 0 -> '', angle 30 -> 'm30')
            suffix = "" if angle == 0 else f"m{angle}"

            # Calculate offset wind direction in radians
            om_rad = np.radians(df["om_wind_direction_10m"] - angle)

            # Column names
            om_cos_col = f"om_wind_direction_10m_cos{suffix}"
            om_sin_col = f"om_wind_direction_10m_sin{suffix}"
            om_speed_cos_col = f"om_wind_speed_10m_x_cos{suffix}_wd10m"
            om_speed_sin_col = f"om_wind_speed_10m_x_sin{suffix}_wd10m"

            # Compute trigonometric features
            df[om_cos_col] = np.cos(om_rad)
            df[om_sin_col] = np.sin(om_rad)

            # Compute vector wind components (Wind Speed * Cosine/Sine)
            df[om_speed_cos_col] = df["om_wind_speed_10m_kt"] * df[om_cos_col]
            df[om_speed_sin_col] = df["om_wind_speed_10m_kt"] * df[om_sin_col]

            for feat in [om_cos_col, om_sin_col, om_speed_cos_col, om_speed_sin_col]:
                if feat not in ENGINEERED_FEATURES:
                    ENGINEERED_FEATURES.append(feat)

        # Generate the same rotated-angle feature family for om_wind_speed_700hPa_kt /
        # om_wind_direction_700hPa (replaces the old single fixed-angle valley
        # projection -- letting feature selection pick the best-fitting angle(s)
        # per category instead of relying on one hand-tuned init_valley_angle).
        for angle in angles:
            suffix = "" if angle == 0 else f"m{angle}"

            syn700_rad = np.radians(df["om_wind_direction_700hPa"] - angle)

            syn700_cos_col = f"om_wind_direction_700hPa_cos{suffix}"
            syn700_sin_col = f"om_wind_direction_700hPa_sin{suffix}"
            syn700_speed_cos_col = f"om_wind_speed_700hPa_kt_x_cos{suffix}_wd700hPa"
            syn700_speed_sin_col = f"om_wind_speed_700hPa_kt_x_sin{suffix}_wd700hPa"

            df[syn700_cos_col] = np.cos(syn700_rad)
            df[syn700_sin_col] = np.sin(syn700_rad)

            df[syn700_speed_cos_col] = df["om_wind_speed_700hPa_kt"] * df[syn700_cos_col]
            df[syn700_speed_sin_col] = df["om_wind_speed_700hPa_kt"] * df[syn700_sin_col]

            for feat in [syn700_cos_col, syn700_sin_col, syn700_speed_cos_col, syn700_speed_sin_col]:
                if feat not in ENGINEERED_FEATURES:
                    ENGINEERED_FEATURES.append(feat)

        # Same rotated-angle feature family for mosmix_ff_kt / mosmix_dd_deg
        # (replaces the old single fixed-angle valley projection for the
        # local MOSMIX surface wind).
        for angle in angles:
            suffix = "" if angle == 0 else f"m{angle}"

            mos_rad = np.radians(df["mosmix_dd_deg"] - angle)

            mos_cos_col = f"mosmix_dd_deg_cos{suffix}"
            mos_sin_col = f"mosmix_dd_deg_sin{suffix}"
            mos_speed_cos_col = f"mosmix_ff_kt_x_cos{suffix}_moswd"
            mos_speed_sin_col = f"mosmix_ff_kt_x_sin{suffix}_moswd"

            df[mos_cos_col] = np.cos(mos_rad)
            df[mos_sin_col] = np.sin(mos_rad)

            df[mos_speed_cos_col] = df["mosmix_ff_kt"] * df[mos_cos_col]
            df[mos_speed_sin_col] = df["mosmix_ff_kt"] * df[mos_sin_col]

            for feat in [mos_cos_col, mos_sin_col, mos_speed_cos_col, mos_speed_sin_col]:
                if feat not in ENGINEERED_FEATURES:
                    ENGINEERED_FEATURES.append(feat)

        # Diurnal-phase candidates (raw hour_int vs. season-robust
        # daylight_frac_elapsed) -- see _add_diurnal_features docstring.
        df = self._add_diurnal_features(df)

        return df
        
    def print_model_summary(self):
        """Displays fitted BayesianRidge parameters, features, and coefficients."""
        print("\n" + "=" * 80)
        print("🤖 BAYESIAN RIDGE MODEL FIT SUMMARY & RETAINED COEFFICIENTS")
        print("=" * 80)
        
        has_models = False

        for target_label, model_dict, feature_dict in [
            ("Mean Wind Speed (ff)", self.bayesian_models, self.category_feature_sets),
            ("Gust Speed (fx1)", self.bayesian_fx1_models, self.category_fx1_feature_sets)
        ]:
            if not model_dict:
                continue
            
            has_models = True
            print(f"\n🎯 Target: {target_label}")
            print("-" * 80)

            for cat, pipeline_model in model_dict.items():
                ridge_model = pipeline_model.named_steps['bayesianridge']
                scaler = pipeline_model.named_steps['standardscaler']
                features = feature_dict.get(cat, [])

                intercept = ridge_model.intercept_
                scaled_coefs = ridge_model.coef_
                unscaled_coefs = scaled_coefs / scaler.scale_

                print(f"📌 Category Regime : [{cat}]")
                print(f"   • Intercept       : {intercept:+.4f}")
                print(f"   • Alpha (Precision): {ridge_model.alpha_:.4f} | Lambda: {ridge_model.lambda_:.4f}")
                print(f"   • Features & Retained Coefficients:")

                for feat, s_c, u_c in zip(features, scaled_coefs, unscaled_coefs):
                    print(f"     - {feat:<28} : {u_c:+.4f} (Scaled Impact: {s_c:+.4f})")
                print()

        if not has_models:
            print("⚠️ No trained Bayesian Ridge models available (using Fallback offsets).")

        print("=" * 80 + "\n")

    def fit_from_database(self):
        """Trains category-specific Bayesian models using calibration logs (11:00 - 17:00)."""
        try:
            if not os.path.exists(self.db_filepath):
                return

            df_db = pd.read_csv(self.db_filepath)
            df_db = df_db.dropna(subset=["mosmix_ff_kt", "obs_speed"]).copy()
            if df_db.empty:
                return

            if 'hour' in df_db.columns:
                df_db['hour_int'] = df_db['hour'].apply(
                    lambda x: int(str(x).split(':')[0]) if pd.notna(x) else np.nan
                )
                df_db = df_db[(df_db['hour_int'] >= self.op_window[0]) & (df_db['hour_int'] <= self.op_window[1])].copy()

            if df_db.empty:
                return

            if "mosmix_fx1_kt" in df_db.columns and "obs_gust" in df_db.columns:
                df_db["delta_fx1_obs_gust"] = df_db["mosmix_fx1_kt"] - df_db["obs_gust"]
                clean_fx_global = df_db.dropna(subset=["delta_fx1_obs_gust"])
                if not clean_fx_global.empty:
                    self.global_fx1_fallback_bias = float(clean_fx_global["delta_fx1_obs_gust"].mean())
                    self.global_fx1_std_bias = float(clean_fx_global["delta_fx1_obs_gust"].std(ddof=1)) if len(clean_fx_global) > 1 else 0.0
                    
            df_db["delta_ff_obs"] = df_db["mosmix_ff_kt"] - df_db["obs_speed"]
            self.global_fallback_bias = float(df_db["delta_ff_obs"].mean())
            self.global_std_bias = float(df_db["delta_ff_obs"].std(ddof=1)) if len(df_db) > 1 else 0.0

            df_db = self._prepare_features(df_db)

            # Fit per category combination using BayesianRidge
            for cat, group in df_db.groupby("classification"):
                group = group.copy()

                # Skip categories that are structurally unreachable at
                # prediction time -- see is_category_rain_locked (single
                # source of truth shared with run_database_analysis's
                # valid_combos filtering).
                if self.is_category_rain_locked(group):
                    continue

                self.category_mean_offsets[cat] = float(group["delta_ff_obs"].mean())
                self.category_std_offsets[cat] = float(group["delta_ff_obs"].std(ddof=1)) if len(group) > 1 else 0.0
                if "delta_fx1_obs_gust" in group.columns:
                    clean_fx_group = group.dropna(subset=["delta_fx1_obs_gust"])
                    if not clean_fx_group.empty:
                        self.category_fx1_mean_offsets[cat] = float(clean_fx_group["delta_fx1_obs_gust"].mean())
                        self.category_fx1_std_offsets[cat] = float(clean_fx_group["delta_fx1_obs_gust"].std(ddof=1)) if len(clean_fx_group) > 1 else 0.0
                
                features_ff = self.category_feature_sets.get(cat)
                if features_ff:
                    clean_g = group.dropna(subset=features_ff + ["delta_ff_obs", "obs_speed"]).copy()
                    if len(clean_g) >= self.min_samples_for_ridge:
                        weights = compute_wingfoil_weights(clean_g["obs_speed"].values, threshold=self.weight_center)
                        m_ff = make_pipeline(StandardScaler(), BayesianRidge(compute_score=True))
                        m_ff.fit(clean_g[features_ff], clean_g["delta_ff_obs"], bayesianridge__sample_weight=weights)
                        self.bayesian_models[cat] = m_ff

                # Fit Gust Model (fx1) -- UNWEIGHTED by design: the sigmoid
                # weighting exists to prioritize accuracy near the ff go/no-go
                # threshold (foil_confirmed_threshold_knots on obs_speed).
                # Gust isn't the decision variable, so fx1 is fit with plain
                # (uniform) sample weight rather than reusing or re-deriving a
                # wind-speed-based weighting scheme for a different target.
                features_fx = self.category_fx1_feature_sets.get(cat)
                if features_fx and "delta_fx1_obs_gust" in group.columns:
                    clean_g_fx = group.dropna(subset=features_fx + ["delta_fx1_obs_gust", "obs_gust"]).copy()
                    if len(clean_g_fx) >= self.min_samples_for_ridge:
                        m_fx = make_pipeline(StandardScaler(), BayesianRidge(compute_score=True))
                        m_fx.fit(clean_g_fx[features_fx], clean_g_fx["delta_fx1_obs_gust"])
                        self.bayesian_fx1_models[cat] = m_fx

            if self.verbose_summary:
                self.print_model_summary()

        except Exception as e:
            print(f"⚠️ Warning: Initialization from calibration DB failed ({e}). Proceeding with baseline fallbacks.")

    @staticmethod
    def base_regime(cat):
        """Strips the ' + Precip(...)' suffix from a composite classification
        string, e.g. 'PartlyCloudy + Precip(Other_Precip)' -> 'PartlyCloudy'.
        Returns cat unchanged if it has no Precip(...) suffix."""
        if cat is None:
            return cat
        idx = cat.find(" + Precip(")
        return cat[:idx] if idx != -1 else cat

    def resolve_category(self, cat, lookup):
        """Resolves `cat` to a key present in `lookup` (a dict keyed by
        classification, e.g. self.bayesian_models or self.category_mean_offsets).

        Composite regimes like 'PartlyCloudy + Precip(Other_Precip)' rarely
        have enough samples of their own to train/calibrate a dedicated
        model (see min_samples_for_ridge), so most of them never appear as
        keys in `lookup` and previously fell straight through to the flat
        global fallback -- silently discarding the base-regime signal
        (cloud cover, soil temp, etc.) even though om_prec_prob no longer
        needs to gate that signal (rain damping already handles precip's
        actual physical effect on wind, see apply_rain_multiplicative_factor).

        Falls back from the full composite string to its base regime
        (stripping ' + Precip(...)') before giving up. Returns None if
        neither the composite nor the base key is present in `lookup`, so
        callers can still fall through to their next-lower fallback tier.
        """
        if cat in lookup:
            return cat
        base = self.base_regime(cat)
        if base != cat and base in lookup:
            return base
        return None

    def apply_rain_multiplicative_factor(self, raw_speed, prec_prob, w_code):

        if raw_speed < 1.2:
            return raw_speed, 1.0

        # Same predicate and threshold classify_precip_type uses to tag a
        # row "Rain" -- see is_rain_damping_code / rain_prob_confirm_threshold.
        # Keeping this as a single shared check is what guarantees a row
        # tagged "Precip(Rain)" always gets damped here, and a row that
        # doesn't clear the bar is tagged "Rain_LowConf" instead of "Rain".
        is_rain_code = self.is_rain_damping_code(w_code)
        prec_prob_known = pd.notna(prec_prob)

        triggers = is_rain_code and prec_prob_known and prec_prob > self.rain_prob_confirm_threshold
        floor = 0.33

        if triggers:
            damping_factor = max(floor, 1.0 - ((1.0 - floor) * (prec_prob / 100.0)))
            return raw_speed * damping_factor, damping_factor

        return raw_speed, 1.0

    def is_category_rain_locked(self, cat_df):
        """True if every row in cat_df would be fully absorbed by
        apply_rain_multiplicative_factor -- i.e. the category is
        structurally unreachable at prediction time.

        In process(), apply_rain_multiplicative_factor is checked before
        resolve_category / the Bayesian models / the offset table: if it
        triggers, its result is used and nothing below it (including any
        model or offset fit for this category) is ever consulted (see
        get_delta_breakdown / process()). Because classify_precip_type now
        gates the "Rain" label on the exact same is_rain_damping_code +
        rain_prob_confirm_threshold check damping uses, a "... +
        Precip(Rain)" category only exists when every one of its rows
        already cleared that trigger -- so fitting or recommending features
        for it produces numbers that are computed correctly but can never
        actually be applied.

        Single source of truth for this check, shared by
        fit_from_database's per-category skip and run_database_analysis's
        valid_combos filtering, so the two can't independently drift the
        way classify_precip_type and the damping factor once did before
        rain_prob_confirm_threshold was unified. Returns False (not
        rain-locked) for an empty cat_df.
        """
        if cat_df.empty:
            return False
        return bool(cat_df.apply(
            lambda r: self.is_rain_damping_code(r.get("om_w_codes"))
            and pd.notna(r.get("om_prec_prob"))
            and r.get("om_prec_prob") > self.rain_prob_confirm_threshold,
            axis=1,
        ).all())

    def get_delta_breakdown(self, row, mos_ff, classification, prec_prob, w_code):
        """
        Calcule la décomposition additive exacte des contributions des facteurs au Δ (correction du vent).
        Retourne un dictionnaire exploitable par la table d'analyse.
        """
        # 1. Cas du facteur multiplicatif de pluie
        _, factor = self.apply_rain_multiplicative_factor(mos_ff, prec_prob, w_code)
        if factor < 1.0:
            return {
                "type": "rain",
                "factor": factor
            }

        # 2. Cas du modèle Bayésien
        cat = classification
        model_cat = self.resolve_category(cat, self.bayesian_models)
        if model_cat is not None:
            pipe = self.bayesian_models[model_cat]
            scaler = pipe.named_steps['standardscaler']
            ridge = pipe.named_steps['bayesianridge']
            features = self.category_feature_sets.get(model_cat, [])

            row_dict = row.to_dict()
            raw_vals = np.array([float(row_dict.get(feat, 0.0)) for feat in features])

            means = scaler.mean_
            scales = scaler.scale_
            scaled_coefs = ridge.coef_
            scaled_intercept = ridge.intercept_

            # Désescalade des coefficients et ajustement de l'intercept
            unscaled_coefs = scaled_coefs / scales
            unscaled_intercept = scaled_intercept - np.sum((scaled_coefs * means) / scales)

            contributions = {
                feat: float(coef * val)
                for feat, coef, val in zip(features, unscaled_coefs, raw_vals)
            }

            return {
                "type": "bayesian",
                "intercept": float(unscaled_intercept),
                "contributions": contributions
            }

        # 3. Cas "pas de modèle" -- catégorie sans Bayesian Ridge fitté.
        # Aligné sur process() : seul un modèle Bayesian Ridge est considéré
        # comme "un modèle disponible" ; l'offset de catégorie fitté n'est
        # plus appliqué, la correction est donc nulle (intercept 0).
        elif self.resolve_category(cat, self.category_mean_offsets) is not None:
            cat = self.resolve_category(cat, self.category_mean_offsets)
            return {
                "type": "no_model",
                "intercept": 0.0,
                "contributions": {}
            }

        # 4. Fallback global -- également nul, même règle que ci-dessus.
        else:
            return {
                "type": "no_model",
                "intercept": 0.0,
                "contributions": {}
            }            
        
    def process(self, df_input):
        """Executes full forecast correction routing for both wind speeds and gusts."""
        df = self._prepare_features(df_input)
        
        corrected_speeds, std_devs = [], []
        corrected_gusts, std_devs_gusts = [], []
        model_used_tags = []
        total_deltas = []
        cat_mean_offsets, cat_std_offsets = [], []
        calib_intercepts = []

        for idx, row in df.iterrows():
            raw_ff = row["mosmix_ff_kt"]
            raw_fx = row.get("mosmix_fx1_kt", raw_ff)
            cat = row["classification"]
            prec_prob = row.get("om_prec_prob", 0.0)
            w_code = row.get("om_w_codes", np.nan)
            row_dict = row.to_dict()

            # Record category calibration statistics
            c_mean_off = self.category_mean_offsets.get(cat, np.nan)
            c_std_off = self.category_std_offsets.get(cat, np.nan)
            cat_mean_offsets.append(c_mean_off)
            cat_std_offsets.append(c_std_off)

            # Record model intercept / delta breakdown details
            breakdown = self.get_delta_breakdown(row, raw_ff, cat, prec_prob, w_code)
            intercept_val = breakdown.get("intercept", np.nan) if isinstance(breakdown, dict) and breakdown.get("type") != "rain" else np.nan
            calib_intercepts.append(intercept_val)

            # =========================================================
            # MEAN SPEED CORRECTION (ff)
            # =========================================================
            rain_damped_ff, factor = self.apply_rain_multiplicative_factor(raw_ff, prec_prob, w_code)
            
            if factor < 1.0:
                final_speed = rain_damped_ff
                pred_std = 0.0
                model_tag = f"Rain Mult ({factor:.2f}x)"

            elif self.resolve_category(cat, self.bayesian_models) is not None:
                ff_model_cat = self.resolve_category(cat, self.bayesian_models)
                features = self.category_feature_sets[ff_model_cat]
                X_feat = pd.DataFrame([[row_dict.get(c, 0.0) for c in features]], columns=features).fillna(0)

                bias_mean, bias_std = self.bayesian_models[ff_model_cat].predict(X_feat, return_std=True)
                final_speed = raw_ff - float(bias_mean[0])
                pred_std = float(bias_std[0])
                model_tag = "Bayesian Ridge" if ff_model_cat == cat else f"Bayesian Ridge (base regime: {ff_model_cat})"

            elif self.resolve_category(cat, self.category_mean_offsets) is not None:
                # No Bayesian model was fit for this category -- treat it as
                # "no model available" and apply no correction rather than
                # falling back to a fitted mean offset (see request: only
                # Bayesian Ridge is trusted to produce a non-zero correction;
                # every lower tier is a 0/0 no-op).
                ff_offset_cat = self.resolve_category(cat, self.category_mean_offsets)
                final_speed = raw_ff
                pred_std = 0.0
                model_tag = "No Model (raw MOS)" if ff_offset_cat == cat else f"No Model (raw MOS, base regime: {ff_offset_cat})"

            else:
                final_speed = raw_ff
                pred_std = 0.0
                model_tag = "No Model (raw MOS, global fallback)"

            # Surface it when a row carries a rain code that didn't clear
            # rain_prob_confirm_threshold (classify_precip_type's
            # "Rain_LowConf") and ends up corrected by something other than
            # the rain multiplicative path -- i.e. precip is present in the
            # input but nothing in the applied correction accounts for it.
            # Without this tag such rows are indistinguishable in
            # correction_engine from ordinary dry-weather corrections (see
            # the 2026-08-02 / 2026-08-15 calibration_db.csv rows this was
            # written for).
            if row.get("precip_type") == "Rain_LowConf" and factor >= 1.0:
                model_tag = f"{model_tag} [low-conf precip, unmodeled]"

            final_speed_unclipped = final_speed
            final_speed = float(np.clip(final_speed, a_min=0.0, a_max=None))

            if self.resolve_category(cat, self.bayesian_fx1_models) is not None:
                fx1_model_cat = self.resolve_category(cat, self.bayesian_fx1_models)
                features_fx = self.category_fx1_feature_sets[fx1_model_cat]
                X_feat_fx = pd.DataFrame([[row_dict.get(c, 0.0) for c in features_fx]], columns=features_fx).fillna(0)

                bias_fx_mean, bias_fx_std = self.bayesian_fx1_models[fx1_model_cat].predict(X_feat_fx, return_std=True)
                final_gust = raw_fx - float(bias_fx_mean[0])
                pred_std_gust = float(bias_fx_std[0])

            elif self.resolve_category(cat, self.category_fx1_mean_offsets) is not None:
                # Same "no model -> no correction" rule as the ff branch
                # above: a fitted mean offset without a Bayesian model is
                # not treated as "a model available" for gust either.
                final_gust = raw_fx
                pred_std_gust = 0.0

            else:
                final_gust = raw_fx
                pred_std_gust = 0.0

            # Ensure corrected gust is physically valid (gust >= mean speed)
            final_gust = float(np.clip(final_gust, a_min=final_speed, a_max=None))

            # Store Results
            corrected_speeds.append(final_speed)
            std_devs.append(pred_std)
            corrected_gusts.append(final_gust)
            std_devs_gusts.append(pred_std_gust)
            model_used_tags.append(model_tag)
            # Use the UNCLIPPED delta so Table 2 (Intercept + Contributions, built from the
            # same underlying model math in get_delta_breakdown) always reconciles with the
            # Total Δ shown here. If final_speed itself was clipped to 0, the wind-speed
            # column still reflects that; only the *delta breakdown* stays un-clipped for
            # consistency between tables.
            total_deltas.append(final_speed_unclipped - raw_ff)

        df["mosmix_ff_corrected_kt"] = corrected_speeds
        df["mosmix_ff_std_kt"] = std_devs
        df["mosmix_fx1_corrected_kt"] = corrected_gusts
        df["mosmix_fx1_std_kt"] = std_devs_gusts
        df["total_corr_kt"] = total_deltas
        df["correction_engine"] = model_used_tags
        df["calib_cat_mean_offset"] = cat_mean_offsets
        df["calib_cat_std_offset"] = cat_std_offsets
        df["calib_global_fallback_bias"] = self.global_fallback_bias
        df["calib_intercept"] = calib_intercepts

        return df
        
    def export_weights_dict(self):
        """Exports fitted weights and scaler parameters with local export date and time."""
        now_local = datetime.now().astimezone()
        export_time_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")

        export_data = {
            "version": "MOSMIX_V22",
            "updated_at": export_time_str,
            "global_fallback_bias": self.global_fallback_bias,
            "global_std_bias": self.global_std_bias,
            "global_fx1_fallback_bias": self.global_fx1_fallback_bias,
            "global_fx1_std_bias": self.global_fx1_std_bias,
            "category_mean_offsets": self.category_mean_offsets,
            "category_std_offsets": self.category_std_offsets,
            "category_fx1_mean_offsets": self.category_fx1_mean_offsets,
            "category_fx1_std_offsets": self.category_fx1_std_offsets,
            "bayesian_models": {},
            "bayesian_fx1_models": {}
        }

        def serialize_pipeline(model_dict, feature_dict, dest):
            for cat, pipe in model_dict.items():
                scaler = pipe.named_steps["standardscaler"]
                ridge = pipe.named_steps["bayesianridge"]
                features = feature_dict.get(cat, [])
                dest[cat] = {
                    "features": features,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "coef": ridge.coef_.tolist(),
                    "intercept": float(ridge.intercept_),
                    "alpha": float(ridge.alpha_),
                    "sigma": ridge.sigma_.tolist()
                }

        serialize_pipeline(self.bayesian_models, self.category_feature_sets, export_data["bayesian_models"])
        serialize_pipeline(self.bayesian_fx1_models, self.category_fx1_feature_sets, export_data["bayesian_fx1_models"])
        return export_data

    def export_weights_to_json(self, filepath="model_weights.json"):
        """
        Exports the fitted pipeline parameters (weights, coefficients, intercepts, and scalers)
        to a JSON file on disk.
        """
        def sanitize_for_json(obj):
            """Recursively converts NumPy data types to native Python types."""
            if isinstance(obj, dict):
                return {str(k): sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return sanitize_for_json(obj.tolist())
            return obj

        weights_data = self.export_weights_dict()
        sanitized_data = sanitize_for_json(weights_data)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(sanitized_data, f, indent=4)
            print(f"✅ Model weights successfully exported to '{filepath}'.")
        except Exception as e:
            print(f"❌ Error exporting model weights to '{filepath}': {e}")
