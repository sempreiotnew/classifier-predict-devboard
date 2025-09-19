import threading
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash

# ------------ SERIAL CONFIG ------------
SERIAL_PORT = "/dev/cu.usbserial-0289722F"
BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0
MAX_BUFFER_LINES = 200_000
# --------------------------------------

serial_lock = threading.Lock()
serial_header = None
serial_buffer = []

# ---------------- SERIAL WORKER ----------------
def serial_worker(port, baud, timeout):
    global serial_header, serial_buffer
    import serial
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=timeout)
            print(f"[serial_reader] Opened {port} @ {baud}")
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    line = raw.decode("latin-1", errors="replace").strip()
                if not line or line.startswith("-"):
                    continue

                # Detect header
                if line.lower().startswith("id,"):
                    new_header = [c.strip() for c in line.split(",")]
                    with serial_lock:
                        if serial_header != new_header:
                            serial_header = new_header
                            serial_buffer.clear()
                            print(f"[serial_reader] NEW header detected, buffer cleared: {serial_header}")
                    continue

                # Append rows after header
                if serial_header is not None:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != len(serial_header):
                        continue
                    row = dict(zip(serial_header, parts))
                    with serial_lock:
                        serial_buffer.append(row)
                        if len(serial_buffer) > MAX_BUFFER_LINES:
                            serial_buffer.pop(0)
                    csv_line = ",".join(str(row[col]) for col in serial_header)
                    print(f"{csv_line}")
        except Exception as e:
            print("[serial_worker] exception:", e)
            time.sleep(1)

# ---------------- LOAD DATA ----------------
def load_data():
    global serial_buffer
    with serial_lock:
        if len(serial_buffer) == 0:
            return pd.DataFrame()
        rows_copy = list(serial_buffer)

    df = pd.DataFrame(rows_copy)
    numeric_cols = ["millis","gas_resistance","temperature","pressure","humidity","gas_index","id"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "b0"]
    if "id" in df.columns:
        df["sensor_key"] = df["id"].astype(str)
    return df

# ---------------- MAKE FIGURE ----------------
def make_grid_figure(df):
    sensors = df["sensor_key"].unique() if "sensor_key" in df.columns else []
    n_sensors = len(sensors)
    cols = 4
    rows = (n_sensors + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type":"scatter"}]*cols for _ in range(rows)],
        subplot_titles=[f"Sensor {s}" for s in sensors],
        shared_xaxes=False,
        shared_yaxes=True,  # mesma escala Y para comparar
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    # Colors exactly like your example
    color_map_forced = {100: "orange", 99: "cyan"}
    parallel_colors = ["red","green","blue","purple","magenta","yellow","lime","teal","pink","brown"]

    # define fixed Y scale for all sensors
    y_min = df["gas_resistance"].min() * 0.8 if not df.empty else 1
    y_max = df["gas_resistance"].max() * 1.2 if not df.empty else 1e8

    for i, sensor in enumerate(sensors):
        r = i // cols + 1
        c = i % cols + 1
        sensor_df = df[df["sensor_key"] == sensor].sort_values("millis")
        for gas_idx in sorted(sensor_df["gas_index"].unique()):
            if pd.isna(gas_idx):
                continue
            try:
                gi = int(gas_idx)
            except ValueError:
                continue
            sub = sensor_df[sensor_df["gas_index"] == gi]
            if sub.empty:
                continue

            if gi in (99,100):
                color = color_map_forced.get(gi,"gray")
                name = f"Forced {gi}"
            else:
                color = parallel_colors[gi % len(parallel_colors)]
                name = f"Step {gi}"

            fig.add_trace(
                go.Scatter(
                    x=sub["millis"],
                    y=sub["gas_resistance"],
                    mode="lines+markers",
                    line=dict(color=color),
                    name=name,
                    showlegend=(i==0),
                    hovertemplate="Gas: %{y:,.0f} Ω<extra></extra>"
                ),
                row=r, col=c
            )
        fig.update_yaxes(title_text="Gas Resistance (Ω, log)", type="log", range=[np.log10(y_min), np.log10(y_max)], row=r, col=c)
        fig.update_xaxes(title_text="Time (ms)", row=r, col=c)

    fig.update_layout(template="plotly_dark", height=350*rows, width=400*cols, hovermode="x unified", title_text="BME688 8-Sensor Dashboard")
    return fig

# ------------------- DASH APP -------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 8-Sensor Dashboard", style={"color":"white","textAlign":"center"}),
    dcc.Graph(id="live-grid-graph"),
    dcc.Interval(id="interval-refresh", interval=500, n_intervals=0)
], style={"backgroundColor":"#111","padding":"20px"})

# ------------------- CALLBACKS -------------------
@app.callback(
    Output("live-grid-graph", "figure"),
    Input("interval-refresh", "n_intervals")
)
def update_graph(n):
    df = load_data()
    if df.empty:
        return go.Figure()
    fig = make_grid_figure(df)
    return fig

# ------------------- MAIN -------------------
if __name__ == "__main__":
    _thread = threading.Thread(target=serial_worker, args=(SERIAL_PORT, BAUDRATE, SERIAL_TIMEOUT), daemon=True)
    _thread.start()
    app.run(debug=True, use_reloader=False)
