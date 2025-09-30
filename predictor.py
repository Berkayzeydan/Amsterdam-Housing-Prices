import pickle

import pandas as pd
import requests



class Predictor:

    def __init__(self):

        with open("models/rf_model.pkl", "rb") as f: self.model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f: self.scaler = pickle.load(f)
        with open("models/train_columns.pkl", "rb") as f: self.TRAIN_COLS = pickle.load(f)
        with open("models/num_order.pkl", "rb") as f: self.NUM_ORDER = pickle.load(f)

    @staticmethod
    def get_lat_lon(address: str):

        params = {
            "q": address,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "countrycodes": "nl"
    }
        headers = {"User-Agent": "your-app-name/1.0 (youremail@example.com)"}

        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()

        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            print(lat, lon)
            return lat, lon
        else:
            print("No results")
            return None


    def predict(self, area, room, out_code, lat, lon):
        X = pd.DataFrame(0, index=[0], columns=self.TRAIN_COLS)
        num_raw = pd.DataFrame([{
            "Area": float(area),
            "Room": int(room),
            "Lat": float(lat),
            "Lon": float(lon),
        }])
        num_scaled = self.scaler.transform(num_raw[self.NUM_ORDER])
        X.loc[0, self.NUM_ORDER] = num_scaled[0]
        o = f"outcode_{out_code}"
        if o in X.columns:
            X.loc[0, o] = 1
        return self.model.predict(X)
