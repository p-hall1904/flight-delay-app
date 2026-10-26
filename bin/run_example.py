"""
Example: train the flight delay model on a small dataset and make a prediction.

This is a standalone version of the notebook that runs from the command line.
It skips S3 and keeps everything local, so no AWS account is needed.

Before running:
  1. pip install requests pandas numpy scikit-learn joblib
  2. Put at least one BTS On-Time Performance CSV for CLT in the data/ folder,
     named like T_ONTIME_REPORTING-January.csv

Run it from the main repo folder:
  python bin/run_example.py
"""

import glob
import io
import os

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Paths are relative to the repo root (the folder above bin/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIGHT_FILES = os.path.join(ROOT, "data", "T_ONTIME_REPORTING*.csv")
MODEL_PATH = os.path.join(ROOT, "models", "delay_model.joblib")

STATION_ID = "KCLT"        # airport code for the live weather API
IEM_STATION_ID = "CLT"     # airport code for the historical weather archive
DELAY_THRESHOLD_MIN = 15   # DOT definition of a delayed flight

FEATURE_COLS = ["temp_c", "dewpoint_c", "wind_speed_kt", "visibility_sm",
                "altimeter_hpa", "low_visibility", "has_gusts", "bad_weather_string"]


# ---------------------------------------------------------------------------
# Weather data
# ---------------------------------------------------------------------------

def collect_weather_data(station_id=STATION_ID, hours_back=3):
    """Get recent METAR observations from the Aviation Weather Center."""
    url = "https://aviationweather.gov/api/data/metar"
    params = {"ids": station_id, "format": "json", "hours": hours_back}
    headers = {"User-Agent": "flight-delay-prediction-example/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    if resp.status_code == 204 or not resp.text.strip():
        return pd.DataFrame()

    df = pd.json_normalize(resp.json())
    rename = {"temp": "temp_c", "dewp": "dewpoint_c", "wspd": "wind_speed_kt",
              "wgst": "wind_gust_kt", "visib": "visibility_sm",
              "altim": "altimeter_hpa", "wxString": "weather_string",
              "obsTime": "obs_time_epoch"}
    df = df.rename(columns=rename)
    # The API leaves out fields that don't apply (like gusts), so add them if missing
    for col in list(rename.values()):
        if col not in df.columns:
            df[col] = np.nan
    # Visibility can come back as "10+", so strip anything that isn't a number
    for col in ["temp_c", "dewpoint_c", "wind_speed_kt", "wind_gust_kt",
                "visibility_sm", "altimeter_hpa"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
            errors="coerce")
    df["obs_time"] = pd.to_datetime(df["obs_time_epoch"], unit="s", utc=True)
    return df


def collect_historical_weather_data(start_date, end_date, station=IEM_STATION_ID):
    """Get older weather from the Iowa Environmental Mesonet archive."""
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": station,
        "data": ["tmpf", "dwpf", "sknt", "gust", "vsby", "alti", "wxcodes"],
        "sts": f"{start_date}T00:00Z",
        "ets": f"{end_date}T00:00Z",
        "tz": "UTC", "format": "onlycomma", "missing": "empty", "trace": "empty",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df

    # Convert Fahrenheit and inches of Hg to Celsius and hPa to match the live API
    df["temp_c"] = (pd.to_numeric(df["tmpf"], errors="coerce") - 32) * 5 / 9
    df["dewpoint_c"] = (pd.to_numeric(df["dwpf"], errors="coerce") - 32) * 5 / 9
    df["altimeter_hpa"] = pd.to_numeric(df["alti"], errors="coerce") * 33.8639
    df["wind_speed_kt"] = pd.to_numeric(df["sknt"], errors="coerce")
    df["wind_gust_kt"] = pd.to_numeric(df["gust"], errors="coerce")
    df["visibility_sm"] = pd.to_numeric(df["vsby"], errors="coerce")
    df["weather_string"] = df["wxcodes"].replace("", np.nan)
    df["obs_time"] = pd.to_datetime(df["valid"], utc=True)
    return df[["obs_time", "temp_c", "dewpoint_c", "wind_speed_kt", "wind_gust_kt",
               "visibility_sm", "altimeter_hpa", "weather_string"]]


# ---------------------------------------------------------------------------
# Flight data and features
# ---------------------------------------------------------------------------

def load_flight_delay_data(pattern=FLIGHT_FILES):
    """Load the BTS CSVs, keep CLT departures, and label each flight delayed or not."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No flight files found. Put a BTS CSV in the data/ folder first.")

    flights = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    flights = flights[flights["ORIGIN"] == "CLT"].copy()

    # Cancelled flights have no delay value, so they get dropped
    flights = flights.dropna(subset=["DEP_DELAY", "CRS_DEP_TIME"])

    # Combine the flight date with the scheduled departure time (HHMM)
    date = pd.to_datetime(flights["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    hhmm = flights["CRS_DEP_TIME"].astype(int).astype(str).str.zfill(4)
    flights["scheduled_dep_ts"] = pd.to_datetime(
        date.dt.strftime("%Y-%m-%d") + " " + hhmm.str[:2] + ":" + hhmm.str[2:],
        errors="coerce", utc=True)
    flights = flights.dropna(subset=["scheduled_dep_ts"])

    flights["delayed"] = (flights["DEP_DELAY"] >= DELAY_THRESHOLD_MIN).astype(int)
    return flights


def add_weather_flags(df):
    """Add the three yes/no features. Used for both training and live data."""
    df["low_visibility"] = (df["visibility_sm"].fillna(10) < 3).astype(int)
    df["has_gusts"] = df["wind_gust_kt"].notna().astype(int)
    df["bad_weather_string"] = df["weather_string"].notna().astype(int)
    return df


def preprocess_data(weather_df, flights_df):
    """Match each flight to the last weather reading before its scheduled departure."""
    weather_df = weather_df.sort_values("obs_time")
    flights_df = flights_df.sort_values("scheduled_dep_ts")

    # Only look at weather before departure (using the actual departure time would leak the answer)
    merged = pd.merge_asof(
        flights_df, weather_df,
        left_on="scheduled_dep_ts", right_on="obs_time",
        direction="backward", tolerance=pd.Timedelta("2h"))
    merged = merged.dropna(subset=["temp_c", "wind_speed_kt"])

    merged = add_weather_flags(merged)
    merged[FEATURE_COLS] = merged[FEATURE_COLS].fillna(merged[FEATURE_COLS].median())
    return merged[FEATURE_COLS + ["delayed"]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def train_model(data):
    """Train a random forest, print the metrics, and save the model."""
    X, y = data[FEATURE_COLS], data["delayed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy: ", round(accuracy_score(y_test, preds), 3))
    print("Precision:", round(precision_score(y_test, preds, zero_division=0), 3))
    print("Recall:   ", round(recall_score(y_test, preds, zero_division=0), 3))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)
    return model


def predict_delay(weather_row, model):
    """Return the probability (0 to 1) that a flight leaving in these conditions is delayed."""
    row = pd.DataFrame([{c: weather_row.get(c, 0) for c in FEATURE_COLS}])
    return float(model.predict_proba(row)[0][1])


# ---------------------------------------------------------------------------
# Run the example
# ---------------------------------------------------------------------------

def main():
    # 1. Load the flights and figure out which dates we need weather for
    print("Loading flight data...")
    flights = load_flight_delay_data()
    print(len(flights), "CLT departures loaded")

    start = (flights["scheduled_dep_ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (flights["scheduled_dep_ts"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 2. Download the historical weather for those same dates
    print(f"Downloading historical weather for {start} to {end}...")
    weather = collect_historical_weather_data(start, end)

    # 3. Build the training table and train
    print("Building features...")
    data = preprocess_data(weather, flights)
    print(len(data), "flights matched to weather,", round(data["delayed"].mean() * 100, 1), "% delayed")

    print("Training model...")
    model = train_model(data)

    # 4. Predict for the weather right now
    print("\nGetting current weather for", STATION_ID, "...")
    live = collect_weather_data()
    if live.empty:
        print("No current weather available right now.")
        return
    live = add_weather_flags(live)
    latest = live.iloc[-1].to_dict()
    print("Chance of a delay right now:", round(predict_delay(latest, model) * 100, 1), "%")


if __name__ == "__main__":
    main()
