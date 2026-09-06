import argparse
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import requests


def calculate_compass_direction(degrees):
    """Convert degrees (0-360) to cardinal compass points."""
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((degrees + 11.25) / 22.5) % 16
    return directions[idx]


def parse_float_safe(val):
    """Safely extracts a numeric float value from strings, dicts, or numbers.

    Returns None (never 0.0) whenever nothing usable can be parsed --
    missing field, empty/non-numeric string, empty dict, or an
    unsupported type. Callers need to be able to tell "genuinely a zero
    reading" apart from "no reading at all"; silently coercing the
    latter to 0.0 used to make gaps in the data look like real calm/zero
    observations on the plots and in the averages.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, dict):
        for k in ["val", "value", "v"]:
            if k in val:
                return parse_float_safe(val[k])
        if val.values():
            return parse_float_safe(next(iter(val.values())))
        return None
    if isinstance(val, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", val)
        if match:
            return float(match.group())
        return None
    return None


def get_ecowitt_data(session, target_date_str, device_id, authorize_code):
    sdate = f"{target_date_str} 00:00"
    edate = f"{target_date_str} 23:59"

    url = "https://www.ecowitt.net/index/get_data"

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://www.ecowitt.net/home/share?authorize={authorize_code}&device_id={device_id}",
        "Origin": "https://www.ecowitt.net",
    }

    payload = {
        "device_id": device_id,
        "authorize": authorize_code,
        "is_list": "0",
        "mode": "0",
        "sdate": sdate,
        "edate": edate,
        "page": "1",
        "sortList": "1|3|51|5|6|33",
        "hideList": "",
    }

    try:
        response = session.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error for {target_date_str}: {e}")
        return None

MS_TO_KMH = 3.6
MS_TO_KNOTS = 1.943844

def extract_weather_timeseries(json_data, speed_unit="kmh"):
    """Parses windspeed, gust, direction, and outdoor temperature from Ecowitt payload.

    convert_units: when True (default), assumes the payload's
    windspeedmph/windgustmph/tempf fields are genuinely mph/°F and
    converts them (speed_unit picks the target speed unit; temperature
    always converts to °C). Set to False for a device whose ecowitt.net
    account already reports these fields pre-converted to the station's
    own display units (some shared devices do this) -- in that case the
    raw numeric values are returned unchanged, with no unit conversion
    applied, since converting an already-converted value corrupts it
    (e.g. a real 15°C reading run through fahrenheit_to_celsius comes out
    as roughly -9°C).

    Any sample that parse_float_safe couldn't parse comes back as None
    in the corresponding output list (never 0.0), so a missing reading
    stays distinguishable from a genuine zero all the way through to the
    plots and any averaging callers do.
    """
    if not json_data or not isinstance(json_data, dict):
        return [], [], [], [], []

    times = json_data.get("times", []) or json_data.get("timeDate", [])
    data_list = json_data.get("list", {})

    if not isinstance(data_list, dict):
        return [], [], [], [], []

    def unwrap_to_list(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            try:
                sorted_keys = sorted(obj.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
                return [obj[k] for k in sorted_keys]
            except Exception:
                return list(obj.values())
        return []

    # 1. Wind speed & gust (native in mph, unless convert_units=False)
    ws_list = data_list.get("wind_speed", {}).get("list", {})
    raw_speeds = unwrap_to_list(ws_list.get("windspeedmph", []))
    raw_gusts = unwrap_to_list(ws_list.get("windgustmph", []))

    # Unit Conversion Factor (from mph)
    conversion_factor = MS_TO_KMH if speed_unit == "kmh" else MS_TO_KNOTS

    def _convert_speed(raw):
        parsed = parse_float_safe(raw)
        return None if parsed is None else parsed * conversion_factor

    speeds = [_convert_speed(s) for s in raw_speeds]
    gusts = [_convert_speed(g) for g in raw_gusts]

    # 2. Wind direction (degrees)
    wd_list = data_list.get("winddir", {}).get("list", {})
    dirs = [parse_float_safe(d) for d in unwrap_to_list(wd_list.get("winddir", []))]

    # 3. Outdoor Temperature (native in °C for this device)
    temp_list = data_list.get("tempf", {}).get("list", {})
    raw_temps = unwrap_to_list(temp_list.get("tempf", []))
    temps_c = [parse_float_safe(t) for t in raw_temps]

    times = unwrap_to_list(times)

    return times, speeds, gusts, dirs, temps_c


def main():
    DEFAULT_DEVICE_ID = "Mzk5bHJCSWxMREpWTEFtKzhoQ1lPUT09"
    DEFAULT_AUTHORIZE = "8E98BV"

    parser = argparse.ArgumentParser(description="Fetch and plot Ecowitt weather data.")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument(
        "--unit",
        type=str,
        choices=["kmh", "knots"],
        default="kmh",
        help="Wind speed unit: 'kmh' (default) or 'knots'",
    )

    args = parser.parse_args()

    if args.date:
        start_dt = datetime.strptime(args.date, "%Y-%m-%d")
        end_dt = start_dt
    elif args.start_date and args.end_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        start_dt = datetime.now()
        end_dt = start_dt

    unit_label = "km/h" if args.unit == "kmh" else "kt"

    session = requests.Session()

    print("Initializing session cookies from Ecowitt...")
    init_url = f"https://www.ecowitt.net/home/share?authorize={DEFAULT_AUTHORIZE}&device_id={DEFAULT_DEVICE_ID}"
    init_headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    session.get(init_url, headers=init_headers)

    all_timestamps, all_speeds, all_gusts, all_directions, all_temps = [], [], [], [], []

    curr_dt = start_dt
    while curr_dt <= end_dt:
        date_str = curr_dt.strftime("%Y-%m-%d")
        print(f"Fetching data for {date_str}...")

        json_data = get_ecowitt_data(session, date_str, DEFAULT_DEVICE_ID, DEFAULT_AUTHORIZE)
        times, speeds, gusts, dirs, temps = extract_weather_timeseries(json_data, speed_unit=args.unit)

        for i, time_val in enumerate(times):
            try:
                if str(time_val).isdigit():
                    dt = datetime.fromtimestamp(int(time_val))
                else:
                    dt = datetime.strptime(str(time_val), "%Y-%m-%d %H:%M")

                # None (missing/unparseable) is kept as None, not
                # coerced to 0.0 -- a real calm reading and a gap in the
                # data must stay distinguishable downstream.
                s_val = speeds[i] if i < len(speeds) else None
                g_val = gusts[i] if i < len(gusts) else None
                d_val = dirs[i] if i < len(dirs) else None
                t_val = temps[i] if i < len(temps) else None

                all_timestamps.append(dt)
                all_speeds.append(s_val)
                all_gusts.append(g_val)
                all_directions.append(d_val)
                all_temps.append(t_val)
            except (ValueError, TypeError):
                continue

        curr_dt += timedelta(days=1)

    if not all_timestamps:
        print("\nNo valid weather data returned for the requested timeframe.")
        return

    # Summary Statistics -- filter out None (missing/unparseable)
    # readings before averaging/min/max so a gap in the data can't drag
    # the numbers toward zero or crash sum()/max() outright.
    valid_speeds = [s for s in all_speeds if s is not None]
    valid_gusts = [g for g in all_gusts if g is not None]
    valid_dirs = [d for d in all_directions if d is not None]
    valid_temps = [t for t in all_temps if t is not None]

    avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else None
    avg_gust = sum(valid_gusts) / len(valid_gusts) if valid_gusts else None
    peak_gust = max(valid_gusts) if valid_gusts else None
    avg_dir = sum(valid_dirs) / len(valid_dirs) if valid_dirs else None
    avg_temp = sum(valid_temps) / len(valid_temps) if valid_temps else None
    min_temp = min(valid_temps) if valid_temps else None
    max_temp = max(valid_temps) if valid_temps else None

    def _fmt(value, fmt):
        return format(value, fmt) if value is not None else "N/A"

    print("\n" + "=" * 40)
    print("      PERIOD WEATHER SUMMARY")
    print("=" * 40)
    speed_str = f"{_fmt(avg_speed, '.2f')} {unit_label}" if avg_speed is not None else "N/A"
    gust_str = f"{_fmt(avg_gust, '.2f')} {unit_label}" if avg_gust is not None else "N/A"
    peak_str = f"{_fmt(peak_gust, '.2f')} {unit_label}" if peak_gust is not None else "N/A"
    print(f"Average Speed : {speed_str}")
    print(f"Average Gust  : {gust_str}")
    print(f"Peak Gust     : {peak_str}")
    if avg_dir is not None:
        print(f"Avg Direction : {avg_dir:.1f}° ({calculate_compass_direction(avg_dir)})")
    else:
        print("Avg Direction : N/A")
    if avg_temp is not None:
        print(f"Average Temp  : {avg_temp:.1f} °C (Min: {_fmt(min_temp, '.1f')} °C, Max: {_fmt(max_temp, '.1f')} °C)")
    else:
        print("Average Temp  : N/A")
    print("=" * 40 + "\n")

    # Matplotlib can't plot Python None, but it does draw a gap (instead
    # of dropping to zero) wherever a series has NaN, so missing/
    # unparseable samples are mapped to NaN only at the plotting
    # boundary -- the None values themselves stay None everywhere else
    # (summary stats above, and any other consumer of this data).
    def _plot_series(values):
        return [float("nan") if v is None else v for v in values]

    plot_speeds = _plot_series(all_speeds)
    plot_gusts = _plot_series(all_gusts)
    plot_directions = _plot_series(all_directions)
    plot_temps = _plot_series(all_temps)

    # 3-Panel Matplotlib Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Wind Speed & Gust
    ax1.plot(all_timestamps, plot_speeds, label=f"Wind Speed ({unit_label})", color="tab:blue", linewidth=1.5)
    ax1.plot(all_timestamps, plot_gusts, label=f"Wind Gust ({unit_label})", color="tab:red", linestyle="--", linewidth=1.2)
    ax1.set_ylabel(f"Speed ({unit_label})")
    title_str = start_dt.strftime("%Y-%m-%d") if start_dt == end_dt else f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
    ax1.set_title(f"Ecowitt Weather Analysis ({title_str})", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Wind Direction
    ax2.scatter(all_timestamps, plot_directions, label="Direction (°)", color="tab:green", s=10, alpha=0.7)
    ax2.set_ylabel("Direction")
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_yticklabels(["0° (N)", "90° (E)", "180° (S)", "270° (W)", "360° (N)"])
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Temperature
    ax3.plot(all_timestamps, plot_temps, label="Air Temperature (°C)", color="tab:orange", linewidth=1.5)
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_xlabel("Time")
    ax3.legend(loc="upper right")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
