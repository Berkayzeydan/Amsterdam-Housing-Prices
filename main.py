from flask import Flask, request, render_template, redirect, url_for
from predictor import *
import re
import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
pre = Predictor()
#app.secret_key = os.environ.get("SECRET_KEY")

def extract_outcode(postal_raw: str) -> str:
    m = re.search(r"\d{4}", (postal_raw or "").upper())
    return m.group(0)


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == "POST":
        street = request.form.get("street")
        postal = request.form.get("postal")
        address = f"{street}, {postal}"
        out_code = extract_outcode(postal)
        house_size = request.form.get("house_size")
        room_num = request.form.get("room_num")
        lat, lon = pre.get_lat_lon(address=address)
        y_pred = pre.predict(area=house_size, room=room_num, lat=lat, lon=lon, out_code=out_code)
        print(f"Predicted price: €{y_pred[0]:,.0f}")
        price = f"€{y_pred[0]:,.0f}"
        return redirect(url_for("result", price=price))

    return render_template("home.html")

@app.route('/result')
def result():
    price = request.args.get("price")
    return render_template("result.html", price=price)


if __name__ == '__main__':
    app.run(debug=True)
