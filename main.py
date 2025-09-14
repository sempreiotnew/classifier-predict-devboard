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

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUDRATE = 115200
CSV_FILE = "data.csv"
MAX_BUFFER_LINES = 50000  # keep last 50k measurements

current_label = None
header = None
start_recording = False
serial_lock = threading.Lock()
serial_buffer = deque(maxlen=MAX_BUFFER_LINES)
label_marks = []  # {"label": str, "start": float, "end": float}

# ---------------- HELPERS ----------------
def _format_cell(v):
    if v is None: return ""
    if isinstance(v, (int,)) or (isinstance(v, float) and abs(v - int(v)) < 1e-9): return str(int(v))
    if isinstance(v, float): return f"{v:.2f}"
    return str(v)

def _label_color(label):
    colors = ["#ef4444","#f97316","#facc15","#22c55e","#3b82f6","#8b5cf6","#ec4899","#0ea5e9"]
    if not label: return "#374151"
    return colors[hash(label) % len(colors)]

# ---------------- SERIAL READER ----------------
def serial_reader():
    global header, start_recording, serial_buffer, current_label
    try: ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    except Exception as e: print(f"Error opening serial port: {e}"); sys.exit(1)
    print(f"[serial_reader] Listening on {SERIAL_PORT}")

    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes: continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line: continue

            if not start_recording and "id" in line.lower() and "index" in line.lower():
                hdr = [c.strip() for c in line.split(",")]
                header = hdr + ["label"]
                if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE)==0:
                    with open(CSV_FILE,"a",newline="") as f: csv.writer(f).writerow(header)
                continue

            if header is None: continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(header)-1: continue
            row = {col_name: parts[i] for i, col_name in enumerate(header[:-1])}

            try:
                for k in ["millis","temperature","pressure","humidity","gas_resistance"]:
                    if k in row: row[k] = float(row[k])
                if "gas_index" in row: row["gas_index"]=int(float(row["gas_index"]))
            except: continue
            if "gas_index" in row and row["gas_index"] in [99,100]: continue

            with serial_lock:
                row["label"] = current_label if current_label else ""
                if "status" not in row: row["status"]="b0"
                serial_buffer.append(row)
                csv_row = [_format_cell(row.get(c,"")) for c in header]
                with open(CSV_FILE,"a",newline="") as f: csv.writer(f).writerow(csv_row)

        except Exception as e: print("[serial_reader] Error:", e)

# ---------------- DASH LABEL SET ----------------
def set_label(new_label):
    global current_label, start_recording, label_marks
    with serial_lock:
        if new_label and new_label.strip():
            now = serial_buffer[-1]["millis"] if serial_buffer else 0
            # close previous label if open
            if current_label and label_marks and "end" not in label_marks[-1]:
                label_marks[-1]["end"] = now
            current_label = new_label.strip()
            start_recording = True
            # start new label
            label_marks.append({"label": current_label, "start": now})
            print(f"[Dash] Label set: {current_label} at {now}ms")

# ---------------- LOAD DATA ----------------
def load_data():
    with serial_lock:
        if not serial_buffer: return pd.DataFrame()
        df = pd.DataFrame(list(serial_buffer))
    for col in ["millis","gas_index","temperature","pressure","humidity","gas_resistance"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if "id" in df.columns and "index" in df.columns: df["sensor_key"] = df["id"].astype(str)+"_S"+df["index"].astype(str)
    return df

# ---------------- MAKE FIGURE ----------------
def make_figure(df, selected_sensor):
    if selected_sensor is None or df.empty: return go.Figure(), []
    d = df[df["sensor_key"]==selected_sensor].sort_values(["millis","gas_index"])
    fig = make_subplots(rows=4, cols=2, specs=[[{"type":"scatter"},{"type":"table"}],
                                                [{"type":"scatter"},{"type":"table"}],
                                                [{"type":"scatter"}, None],
                                                [{"type":"scatter"}, None]],
                        column_widths=[0.65,0.35], row_heights=[0.5,0.17,0.17,0.16],
                        shared_xaxes=True, vertical_spacing=0.05)
    colors = ["red","green","blue","purple","magenta","yellow","lime","teal","pink","brown"]
    for gi in sorted(d["gas_index"].dropna().unique()):
        sub = d[d["gas_index"]==gi]
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub["millis"], y=sub["gas_resistance"], mode="lines+markers",
                                 name=f"Step {int(gi)}", line=dict(color=colors[int(gi)%len(colors)])), row=1,col=1)
    # Other traces
    fig.add_trace(go.Scatter(x=d["millis"], y=d["temperature"], mode="lines+markers", name="Temperature"), row=2,col=1)
    fig.add_trace(go.Scatter(x=d["millis"], y=d["pressure"], mode="lines+markers", name="Pressure"), row=3,col=1)
    fig.add_trace(go.Scatter(x=d["millis"], y=d["humidity"], mode="lines+markers", name="Humidity"), row=4,col=1)
    # ---------------- Label markers and faded area ----------------
    for mark in label_marks:
        start = mark["start"]
        end = mark.get("end", d["millis"].max())
        color = _label_color(mark["label"])
        # vertical markers
        fig.add_vline(x=start, line=dict(color=color, width=3, dash="dash"), row=1,col=1)
        if "end" in mark: fig.add_vline(x=end, line=dict(color=color, width=3, dash="dash"), row=1,col=1)
        # faded area
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.2, line_width=0, row=1,col=1)
    # Table
    table_cols = ["id","index","gas_resistance","status","gas_index"]
    present_cols = [c for c in table_cols if c in d.columns]
    d_sorted = d.sort_values("millis", ascending=False)
    fig.add_trace(go.Table(header=dict(values=[f"<b>{c}</b>" for c in present_cols], fill_color="#111", font=dict(color="white")),
                           cells=dict(values=[d_sorted[c] for c in present_cols], fill_color="#1f2937", font=dict(color="white"))), row=1,col=2)
    # Layout
    fig.update_layout(template="plotly_dark", hovermode="x unified", height=1100, width=1500, title=f"Sensor: {selected_sensor}")
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", row=1,col=1, type="log")
    fig.update_yaxes(title_text="Temperature (°C)", row=2,col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=3,col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=4,col=1)
    fig.update_xaxes(title_text="Time (ms)", row=4,col=1, rangeslider_visible=True, rangeslider_thickness=0.1)
    return fig, df["sensor_key"].unique().tolist()

# ---------------- DASH APP ----------------
app = Dash(__name__)
app.layout = html.Div([
    html.H2("BME688 Live Dashboard", style={"color":"white","textAlign":"center"}),
    html.Div([dcc.Input(id="label-input", type="text", placeholder="Enter new label"),
              html.Button("Set Label", id="set-label-btn", n_clicks=0),
              html.Div(id="current-label-display", style={"color":"white","marginTop":"10px","fontSize":"22px","fontWeight":"bold","padding":"10px","borderRadius":"8px","textAlign":"center"})],
             style={"textAlign":"center","marginBottom":"20px"}),
    dcc.Dropdown(id="sensor-dropdown", options=[], value=None, style={"width":"400px","margin":"auto"}),
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-refresh", interval=500, n_intervals=0)
], style={"backgroundColor":"#111","padding":"20px"})

# ---------------- DASH CALLBACKS ----------------
@app.callback(Output("current-label-display","children"), Output("current-label-display","style"),
              Input("set-label-btn","n_clicks"), State("label-input","value"))
def dash_set_label(n, value):
    if n and value: set_label(value)
    style = {"color":"white","marginTop":"10px","fontSize":"22px","fontWeight":"bold","padding":"10px","borderRadius":"8px","textAlign":"center",
             "backgroundColor": _label_color(current_label)}
    return f"Current Label: {current_label if current_label else 'None'}", style

@app.callback([Output("sensor-dropdown","options"), Output("sensor-dropdown","value")],
              Input("interval-refresh","n_intervals"), State("sensor-dropdown","value"))
def update_sensors(n,current_value):
    df = load_data()
    if df.empty or "sensor_key" not in df.columns: return [],None
    sensors = df["sensor_key"].unique()
    value = current_value if current_value in sensors else sensors[0]
    return [{"label": s,"value": s} for s in sensors], value

@app.callback(Output("live-graph","figure"), [Input("interval-refresh","n_intervals"), Input("sensor-dropdown","value")])
def update_graph(n, selected_sensor):
    df = load_data()
    fig,_ = make_figure(df, selected_sensor)
    return fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, use_reloader=False)
