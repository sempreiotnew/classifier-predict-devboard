import os
import csv
import serial
import threading
import sys
from collections import deque
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import joblib

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUDRATE = 115200
CSV_FILE = "data.csv"
MAX_BUFFER_LINES = 50000  # keep last 50k measurements
MODEL_FILE = "gas_model.pkl"

# ---------------- GLOBAL STATE ----------------
header = None
start_recording = False
serial_lock = threading.Lock()
serial_buffer = deque(maxlen=MAX_BUFFER_LINES)
current_prediction = None  # 🔹 store latest prediction

# Load trained model
clf, le = joblib.load(MODEL_FILE)

# helper to format CSV cells
def _format_cell(v):
    if v is None:
        return ""
    if isinstance(v, (int,)) or (isinstance(v, float) and abs(v - int(v)) < 1e-9):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

# Prediction helper
def predict_label(data_dict):
    df = pd.DataFrame([data_dict])
    pred_encoded = clf.predict(df)
    pred_label = le.inverse_transform(pred_encoded)
    return pred_label[0]

# ---------------- SERIAL READER ----------------
def serial_reader():
    global header, start_recording, serial_buffer, current_prediction

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

            # detect header
            if not start_recording and "id" in line.lower() and "index" in line.lower():
                hdr = [c.strip() for c in line.split(",")]
                header = hdr
                start_recording = True
                print("[serial_reader] Header detected:", header)
                continue

            if start_recording:
                parts = [p.strip() for p in line.split(",")]
                if header is None or len(parts) < len(header):
                    continue

                row = {}
                for i, col_name in enumerate(header):
                    row[col_name] = parts[i]

                try:
                    gas_index = int(float(row["gas_index"]))
                    if gas_index in [99, 100]:
                        continue

                    data_dict = {
                        'millis': float(row["millis"]),
                        'gas_index': gas_index,
                        'mes_index': int(float(row["mes_index"])),
                        'temperature': float(row["temperature"]),
                        'pressure': float(row["pressure"]),
                        'humidity': float(row["humidity"]),
                        'gas_resistance': float(row["gas_resistance"])
                    }

                    # 🔹 Predict and update global state
                    current_prediction = predict_label(data_dict)

                except (ValueError, KeyError):
                    continue

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
        df = pd.DataFrame(list(serial_buffer))

    if "id" in df.columns and "index" in df.columns:
        df["sensor_key"] = df["id"].astype(str) + "_S" + df["index"].astype(str)

    return df

# ---------------- MAKE FIGURE ----------------
def make_figure(df, selected_sensor):
    if selected_sensor is None or df.empty:
        return go.Figure(), []

    d = df.sort_values(["millis", "gas_index"])

    fig = make_subplots(
        rows=4, cols=1,
        specs=[[{"type":"scatter"}],
               [{"type":"scatter"}],
               [{"type":"scatter"}],
               [{"type":"scatter"}]],
        shared_xaxes=True, vertical_spacing=0.05
    )

    if not d.empty and "gas_index" in d.columns:
        colors = ["red","green","blue","purple","magenta","yellow","lime","teal","pink","brown"]
        unique_gi = d["gas_index"].dropna().unique()
        for gi in sorted(unique_gi):
            sub = d[d["gas_index"] == gi]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["millis"],
                y=sub["gas_resistance"],
                mode="lines+markers",
                name=f"Step {int(gi)}",
                line=dict(color=colors[int(gi) % len(colors)]),
            ), row=1, col=1)

    if "temperature" in d.columns:
        fig.add_trace(go.Scatter(x=d["millis"], y=d["temperature"], mode="lines+markers", name="Temperature (°C)"), row=2, col=1)
    if "pressure" in d.columns:
        fig.add_trace(go.Scatter(x=d["millis"], y=d["pressure"], mode="lines+markers", name="Pressure (Pa)"), row=3, col=1)
    if "humidity" in d.columns:
        fig.add_trace(go.Scatter(x=d["millis"], y=d["humidity"], mode="lines+markers", name="Humidity (%)"), row=4, col=1)

    fig.update_layout(template="plotly_dark", hovermode="x unified",
                      height=900, width=1200, title="Sensor Data + Prediction")
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=3, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=4, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=4, col=1, rangeslider_visible=True, rangeslider_thickness=0.1)

    return fig, ["virtual_sensor"]

# ---------------- DASH APP ----------------
app = Dash(__name__)
app.layout = html.Div([
    html.H2("BME688 Live Prediction Dashboard", style={"color":"white","textAlign":"center"}),

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
    [Output("current-prediction-display","children"), Output("live-graph","figure")],
    Input("interval-refresh","n_intervals")
)
def update_dashboard(n):
    df = load_data()
    fig, _ = make_figure(df, "virtual_sensor")
    return f"Current Prediction: {current_prediction if current_prediction else 'None'}", fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, use_reloader=False)
