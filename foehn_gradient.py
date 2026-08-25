#!/usr/bin/env python3
import io
import json
import os
import requests
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

CONFIG_FILE = "config.json"

# Fallback si config.json est absent ou ne contient pas 'foehn_stations'
DEFAULT_STATIONS = {
    "Zurich": "06660",
    "Lugano": "06770"
}

DEFAULT_TIMEZONE = "Europe/Zurich"


def load_foehn_stations():
    """Charge la paire de stations MOSMIX (DWD) utilisée pour le gradient de Foehn
    depuis config.json (clé 'foehn_stations'), avec repli sur DEFAULT_STATIONS."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            stations = user_config.get("foehn_stations")
            if stations and len(stations) >= 2:
                return stations
        except Exception as e:
            print(f"[⚠️ Warning] Erreur lors de la lecture de {CONFIG_FILE} : {e}")
    return DEFAULT_STATIONS


def load_timezone():
    """Charge le fuseau horaire local depuis config.json (clé 'settings.timezone'),
    avec repli sur DEFAULT_TIMEZONE. target_date (passé par les appelants) est
    toujours une date Europe/Zurich -- today_str doit être calculé dans le même
    fuseau, sinon un serveur hébergé en UTC peut décaler la comparaison
    'target_date <= today_str' d'un jour près de minuit."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            tz_name = user_config.get("settings", {}).get("timezone")
            if tz_name:
                return tz_name
        except Exception as e:
            print(f"[⚠️ Warning] Erreur lors de la lecture de {CONFIG_FILE} : {e}")
    return DEFAULT_TIMEZONE


# Configuration des stations MOSMIX (DWD) et du fuseau horaire local
STATIONS = load_foehn_stations()
LOCAL_TIMEZONE = load_timezone()

def clean_namespaces(root):
    """ Supprime récursivement tous les espaces de noms pour faciliter la recherche de balises """
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        for key in list(elem.attrib.keys()):
            if '}' in key:
                new_key = key.split('}', 1)[1]
                elem.attrib[new_key] = elem.attrib.pop(key)
    return root


def fetch_station_data(station_id, station_name, target_date):
    """
    Télécharge MOSMIX_L :
    - Si target_date < aujourd'hui : utilise le fichier de 09h00.
    - Si target_date == aujourd'hui : utilise le fichier 'latest' (03h ou 09h).
    """
    today_str = datetime.now(ZoneInfo(LOCAL_TIMEZONE)).strftime("%Y-%m-%d")
    date_part = target_date.replace("-", "")
    
    if target_date <= today_str:
        # On utilise le premier fichier de la journée de 03h:
        # Les données commencent à 6h, assez tôt pour les prédictions de vent à 10h
        # Note: DWD utilise souvent le format 'YYYYMMDD03' pour les archives
        kmz_filename = f"MOSMIX_L_{date_part}03_{station_id}.kmz"
        base_url = f"https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations/{station_id}/kml/"
    elif target_date > today_str:
        # on utilise le fichier LATEST à partir du lendemain
        kmz_filename = f"MOSMIX_L_LATEST_{station_id}.kmz"
        base_url = f"https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations/{station_id}/kml/"    
        
    url = base_url + kmz_filename
    print(f"MOSMIX FILE: {url}\n")
       
    print(f"-> Tentative de récupération pour {station_name} le {target_date}...")
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        res.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(res.content)) as zip_ref:
            kml_name = next(name for name in zip_ref.namelist() if name.endswith('.kml'))
            kml_content = zip_ref.read(kml_name)
    except Exception as e:
        print(f"   ❌ Erreur lors du téléchargement/extraction pour {station_name} : {e}")
        return None
        
    try:
        #print(f"DEBUG: Fichier {kmz_filename} chargé. Contenu XML détecté : {len(kml_content)} octets.")
        root = clean_namespaces(ET.fromstring(kml_content))

        # 1. Extraction des pas de temps
        timesteps = []
        timesteps_element = root.find('.//ForecastTimeSteps')
        if timesteps_element is not None:
            for ts in timesteps_element.findall('TimeStep'):
                timesteps.append(ts.text)
        
        if not timesteps:
            return None

        # 2. Extraction des valeurs de pression (PPPP)
        pressure_values = []
        forecast_elements = root.findall('.//Forecast')
        for elem in forecast_elements:
            if elem.get("elementName") == "PPPP":
                value_element = elem.find('value')
                if value_element is not None and value_element.text:
                    raw_values = value_element.text.split()
                    for val in raw_values:
                        val = val.strip()
                        pressure_values.append(None if val == "-" or not val else float(val) / 100.0)
                break

        return dict(zip(timesteps, pressure_values))

    except Exception as e:
        print(f"   ❌ Erreur lors du parsing pour {station_name} : {e}")
        return None


def get_combined_data_foehn_gradient(target_date, stations=None):
    """ Sous-routine : Centralise la lecture et calcule le dp entre les stations.

    stations: dict optionnel {"Zurich": station_id, "Lugano": station_id}.
    Si non fourni, utilise la paire chargée depuis config.json (ou le repli par défaut).
    """
    stations = stations or STATIONS
    dict_zh = fetch_station_data(stations["Zurich"], "Zurich", target_date)
    dict_lu = fetch_station_data(stations["Lugano"], "Lugano", target_date)
    
    if not dict_zh or not dict_lu:
        print("\n❌ Impossible de récupérer les données des deux stations.")
        return []

    common_timestamps = sorted(list(set(dict_zh.keys()).intersection(set(dict_lu.keys()))))
    combined_records = []

    #print("\n" + "="*100)
    #print("date, heure, pression Zurich, pression Lugano, dp (ZH - LU) [Nordfoehn], dp (LU - ZH)")
    #print("="*100)

    for ts in common_timestamps:
        p_zh = dict_zh.get(ts)
        p_lu = dict_lu.get(ts)
        
        if p_zh is not None and p_lu is not None:
            # 1. On convertit la chaîne ISO en Timestamp Pandas
            dt_utc = pd.to_datetime(ts)
            if dt_utc.tz is None:
                dt_utc = dt_utc.tz_localize('UTC')
                
            # 2. Conversion vers l'heure suisse et retrait du flag TZ (timezone-naive)
            dt = dt_utc.tz_convert('Europe/Zurich').tz_localize(None)
            
            dp_zh_lu = p_zh - p_lu
            dp_lu_zh = p_lu - p_zh
            
            #print(f"{dt.strftime('%Y-%m-%d')}, {dt.strftime('%H:%M')}, {p_zh:.1f} hPa, {p_lu:.1f} hPa, {dp_zh_lu:+.1f} hPa, {dp_lu_zh:+.1f} hPa")
            
            combined_records.append({
                'datetime': dt,
                'p_zh': p_zh,
                'p_lu': p_lu,
                'dp_foehn': dp_zh_lu  
            })
            
    return combined_records


def plot_foehn_gradient(records):
    """ Fonction Plot Corrigée : Alignée sur la physique réelle ZH - LU """
    if not records:
        return
        
    dates = [r['datetime'] for r in records]
    dp = [r['dp_foehn'] for r in records] # Contient p_lu - p_zh
    
    plt.figure(figsize=(12, 6))
    
    # Label corrigé : c'est bien Zurich - Lugano
    plt.plot(dates, dp, label=r'$\Delta p$ Zurich - Lugano (Nordfoehn si > 0)', color='#2c3e50', linewidth=2.5, zorder=4)
    plt.axhline(0, color='black', linestyle='-', linewidth=1, zorder=2)
    
    # Seuils inversés pour correspondre à la réalité physique : Positif = Nord / Négatif = Sud
    plt.axhline(4, color='#e74c3c', linestyle='--', alpha=0.8, label='Seuil Nordfoehn (+4 hPa)', zorder=2)

    # Surlignage des phases actives
    plt.fill_between(dates, dp, 4, where=[val > 4 for val in dp], facecolor='#e74c3c', alpha=0.25, label='Nordfoehn actif')

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    
    ax.yaxis.set_inverted(True)
    
    plt.grid(True, which='major', linestyle='-', alpha=0.5, zorder=1)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2, zorder=1)

    plt.title("Prévision du Gradient de Pression Alpin (Zurich - Lugano) — Alignement Davos", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Date (Ticks à Minuit)", fontsize=11, labelpad=10)
    plt.ylabel(r"$\Delta p$ (hPa)", fontsize=11)
    plt.legend(loc='upper left', framealpha=0.95)
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


# --- Exécution principale ---
if __name__ == "__main__":
    print("=== DEBUT DE L'EXTRACTION MOSMIX ===")
    donnees = get_combined_data_foehn_gradient(datetime.now(ZoneInfo(LOCAL_TIMEZONE)).strftime('%Y-%m-%d'))
    if donnees:
        plot_foehn_gradient(donnees)
