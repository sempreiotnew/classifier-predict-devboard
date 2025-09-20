import os
import serial
import threading
import sys
from collections import deque
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import joblib
from datetime import datetime

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUDRATE = 115200
MAX_BUFFER_LINES = 50000  # keep last 50k measurements
MODEL_FILE = "gas_model.pkl"

# ---------------- GLOBAL STATE ----------------
header = ["id","index","millis","gas_index","mes_index",
          "temperature","pressure","humidity","gas_resistance","status"]
serial_lock = threading.Lock()
serial_buffer = deque(maxlen=MAX_BUFFER_LINES)
current_prediction = None
start_time = datetime.now()  # reference for millis → timestamp conversion

# Load trained model
clf, le = joblib.load(MODEL_FILE)

# ---------------- Prediction helper ----------------
def predict_label(data_dict):
    # only features for the model
    features = ['temperature','gas_resistance']
    df = pd.DataFrame([{f: data_dict[f] for f in features}])
    pred_encoded = clf.predict(df)
    pred_label = le.inverse_transform(pred_encoded)
    return pred_label[0]

# ---------------- SERIAL READER ----------------
def serial_reader():
    global current_prediction

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    except Exception as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        sys.exit(1)

    print(f"[serial_reader] Listening on {SERIAL_PORT} @ {BAUDRATE} baud")

    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # Strip "[serial]" prefix if present
            if line.startswith("[serial]"):
                line = line.replace("[serial]", "", 1).strip()

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(header):
                continue

            row = {col: parts[i] for i, col in enumerate(header)}

            # accept only rows with status == b0
            if row.get("status", "").lower() != "b0":
                continue

            try:
                data_dict = {
                    "id": row["id"],
                    "index": int(float(row["index"])),
                    "millis": float(row["millis"]),
                    "gas_index": int(float(row["gas_index"])),
                    "mes_index": int(float(row["mes_index"])),
                    "temperature": float(row["temperature"]),
                    "pressure": float(row["pressure"]),
                    "humidity": float(row["humidity"]),
                    "gas_resistance": float(row["gas_resistance"]),
                    "status": row["status"]
                }
            except (ValueError, KeyError):
                continue

            # 🔹 Predict
            current_prediction = predict_label(data_dict)
            data_dict["prediction"] = current_prediction

            with serial_lock:
                serial_buffer.append(data_dict)

            print(f"[DEBUG] Data={data_dict} → Prediction={current_prediction}")

        except Exception as e:
            print("[serial_reader] Error:", e)

# ---------------- LOAD DATA ----------------
def load_data():
    with serial_lock:
        if len(serial_buffer) == 0:
            return pd.DataFrame()
        return pd.DataFrame(list(serial_buffer))

# ---------------- MAKE FIGURE ----------------
def make_figure(df):
    if df.empty:
        return go.Figure()

    d = df.sort_values(["millis", "gas_index"])
    d["timestamp"] = start_time + pd.to_timedelta(d["millis"], unit="ms")

    fig = make_subplots(
        rows=4, cols=1,
        specs=[[{"type":"scatter"}],
               [{"type":"scatter"}],
               [{"type":"scatter"}],
               [{"type":"scatter"}]],
        shared_xaxes=True, vertical_spacing=0.05
    )

    # Gas resistance per gas_index
    if "gas_index" in d.columns:
        colors = ["red","green","blue","purple","magenta",
                  "yellow","lime","teal","pink","brown"]
        for gi in sorted(d["gas_index"].dropna().unique()):
            sub = d[d["gas_index"] == gi]
            fig.add_trace(go.Scatter(
                x=sub["timestamp"],
                y=sub["gas_resistance"],
                mode="lines+markers",
                name=f"Step {int(gi)}",
                line=dict(color=colors[int(gi) % len(colors)])
            ), row=1, col=1)

    # 🔹 Add prediction text in bottom-right of Gas Resistance subplot
    if current_prediction:
        fig.add_annotation(
            text=f"Prediction: {current_prediction}",
            xref="x1", yref="y1",   # bind to Gas Resistance axes
            x=d["timestamp"].max(),  # right edge of x-axis
            y=d["gas_resistance"].min(),  # bottom of y-axis
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(16,185,129,0.7)",
            borderpad=4
        )

    # Temperature
    if "temperature" in d.columns:
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["temperature"],
                                 mode="lines+markers", name="Temperature (°C)"),
                      row=2, col=1)
    # Pressure
    if "pressure" in d.columns:
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["pressure"],
                                 mode="lines+markers", name="Pressure (Pa)"),
                      row=3, col=1)
    # Humidity
    if "humidity" in d.columns:
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["humidity"],
                                 mode="lines+markers", name="Humidity (%)"),
                      row=4, col=1)

    fig.update_layout(template="plotly_dark", hovermode="x unified",
                      height=900, width=1200,
                      title="Sensor Data + Prediction")
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=3, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1,
                     rangeslider_visible=True, rangeslider_thickness=0.1)

    return fig


# ---------------- DASH APP ----------------
app = Dash(__name__)
app.layout = html.Div([
    html.H2("BME688 Live Prediction Dashboard",
            style={"color":"white","textAlign":"center"}),

    # 🔹 Prediction display
    html.Div(id="current-prediction-display",
             style={"backgroundColor":"#10b981","color":"white","padding":"10px",
                    "borderRadius":"8px","textAlign":"center","margin":"10px auto",
                    "width":"400px","fontSize":"20px"}),

    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-refresh", interval=500, n_intervals=0)
], style={"backgroundColor":"#111","padding":"20px"})

# ---------------- CALLBACKS ----------------
@app.callback(
    [Output("current-prediction-display","children"),
     Output("live-graph","figure")],
    Input("interval-refresh","n_intervals")
)
def update_dashboard(n):
    df = load_data()
    fig = make_figure(df)
    return f"{current_prediction if current_prediction else 'None'}", fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, use_reloader=False)
