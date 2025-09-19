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
# We'll initialize header to the default (ensures CSV has header). If serial sends its own header later, we'll skip it.
DEFAULT_HEADER = ["id","index","millis","gas_index","mes_index",
                  "temperature","pressure","humidity","gas_resistance","status"]
header = DEFAULT_HEADER + ["label"]

serial_lock = threading.Lock()
serial_buffer = deque(maxlen=MAX_BUFFER_LINES)
label_marks = []  # {"label": str, "start": float, "end": float}

# ensure CSV has header row immediately (file MUST have header)
if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(header)

# ---------------- HELPERS ----------------
def _format_cell(v):
    if v is None:
        return ""
    if isinstance(v, (int,)) or (isinstance(v, float) and abs(v - int(v)) < 1e-9):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

def _label_color(label):
    colors = ["#ef4444","#f97316","#facc15","#22c55e","#3b82f6","#8b5cf6","#ec4899","#0ea5e9"]
    if not label:
        return "#374151"
    return colors[hash(label) % len(colors)]

# ---------------- SERIAL READER ----------------
def serial_reader():
    global header, serial_buffer, current_label
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    except Exception as e:
        print(f"Error opening serial port: {e}")
        sys.exit(1)
    print(f"[serial_reader] Listening on {SERIAL_PORT}")

    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # print raw serial line exactly as received (matching your examples)
            print(f"[serial] {line}")

            # If the device sends a header line (like "id,index,...") just skip it.
            # We already wrote a default header to the CSV at startup.
            low = line.lower()
            if low.startswith("id,") or low.startswith("millis,"):
                # ignore serial-sent header line
                continue

            parts = [p.strip() for p in line.split(",")]
            # parts must match number of data columns (header without label)
            if len(parts) != len(header) - 1:
                # unexpected format -> skip
                continue

            row = {col_name: parts[i] for i, col_name in enumerate(header[:-1])}

            # convert numeric fields safely where appropriate
            for k in ["id","index","millis","gas_index","mes_index","temperature","pressure","humidity","gas_resistance"]:
                if k in row:
                    try:
                        row[k] = float(row[k])
                    except:
                        row[k] = None

            # # ONLY accept rows whose status is b0 (case-insensitive)
            status_val = str(row.get("status","")).strip().lower()
            if status_val != "b0":
                # we still printed the raw line above, but we drop it here.
                continue

            with serial_lock:
                # add label (empty string if none yet)
                row["label"] = current_label if current_label else ""
                # ensure status present
                if "status" not in row or row["status"] is None:
                    row["status"] = "b0"
                # append to buffer (used by plotting)
                serial_buffer.append(row)

                # write to CSV only when a label is set (prevents messy data)
                if current_label:
                    csv_row = [_format_cell(row.get(c, "")) for c in header]
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(csv_row)

        except Exception as e:
            print("[serial_reader] Error:", e)

# ---------------- DASH LABEL SET ----------------
def set_label(new_label):
    global current_label, label_marks
    with serial_lock:
        if new_label and new_label.strip():
            now = serial_buffer[-1]["millis"] if serial_buffer else 0
            # close previous label if it didn't have an end
            if current_label and label_marks and "end" not in label_marks[-1]:
                label_marks[-1]["end"] = now
            current_label = new_label.strip()
            label_marks.append({"label": current_label, "start": now})
            print(f"[Dash] Label set: {current_label} at {now}ms")

# ---------------- LOAD DATA ----------------
def load_data():
    with serial_lock:
        if not serial_buffer:
            return pd.DataFrame()
        df = pd.DataFrame(list(serial_buffer))
    # numeric conversion
    for col in ["id","index","millis","gas_index","mes_index","temperature","pressure","humidity","gas_resistance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # generate sensor_key from id
    if "id" in df.columns:
        # keep as integer-like string
        df["sensor_key"] = df["id"].astype("Int64").astype(str)
    else:
        df["sensor_key"] = "Sensor_1"
    return df

# ---------------- MAKE FIGURE ----------------
def make_figure(df, selected_sensor):
    if selected_sensor is None or df.empty:
        return go.Figure(), []
    # sort and select
    d = df[df["sensor_key"] == selected_sensor].sort_values(["millis","gas_index"])
    fig = make_subplots(rows=4, cols=2,
                        specs=[[{"type":"scatter"},{"type":"table"}],
                               [{"type":"scatter"},{"type":"table"}],
                               [{"type":"scatter"}, None],
                               [{"type":"scatter"}, None]],
                        column_widths=[0.65,0.35], row_heights=[0.5,0.17,0.17,0.16],
                        shared_xaxes=True, vertical_spacing=0.05)
    colors = ["red","green","blue","purple","magenta","yellow","lime","teal","pink","brown"]

    # gas_resistance traces per gas_index
    if "gas_index" in d.columns:
        for gi in sorted(d["gas_index"].dropna().unique()):
            sub = d[d["gas_index"] == gi]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub["millis"], y=sub["gas_resistance"], mode="lines+markers",
                                     name=f"Step {int(gi)}", line=dict(color=colors[int(gi) % len(colors)])),
                          row=1, col=1)

    # Temperature / Pressure / Humidity traces
    if "temperature" in d.columns:
        fig.add_trace(go.Scatter(x=d["millis"], y=d["temperature"], mode="lines+markers", name="Temperature"), row=2, col=1)
    if "pressure" in d.columns:
        fig.add_trace(go.Scatter(x=d["millis"], y=d["pressure"], mode="lines+markers", name="Pressure"), row=3, col=1)
    if "humidity" in d.columns:
        fig.add_trace(go.Scatter(x=d["millis"], y=d["humidity"], mode="lines+markers", name="Humidity"), row=4, col=1)

    # Label visuals: vlines, vrect and text annotations (stacked to avoid overlap)
    # We'll compute a base y and stack labels by index to avoid overlaps.
    for idx, mark in enumerate(label_marks):
        start = mark["start"]
        end = mark.get("end", None)
        # if end missing, place end at current max millis of this sensor (so rect shows)
        if end is None:
            try:
                end = float(d["millis"].max())
            except:
                end = start
        color = _label_color(mark["label"])

        # vertical lines and faded rect
        fig.add_vline(x=start, line=dict(color=color, width=3, dash="dash"), row=1, col=1)
        if end is not None:
            fig.add_vline(x=end, line=dict(color=color, width=3, dash="dash"), row=1, col=1)
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.15, line_width=0, row=1, col=1)

        # determine base y for annotation (ensure > 0 for log axis)
        if "gas_resistance" in d.columns and not d["gas_resistance"].dropna().empty:
            ymax = float(d["gas_resistance"].dropna().max())
            if ymax <= 0:
                ymax = 1.0
            base_y = ymax * 1.05
        elif "temperature" in d.columns and not d["temperature"].dropna().empty:
            base_y = float(d["temperature"].dropna().max()) + 1.0
        else:
            base_y = 1.0

        # stack by index to reduce overlap (use multiplicative offset)
        offset_factor = 1.0 + 0.06 * (idx % 8)
        y_val = base_y * offset_factor

        # annotation at the start (shows label name)
        fig.add_annotation(x=start, y=y_val,
                           text=mark["label"],
                           showarrow=True, arrowhead=2, ax=0, ay=-30,
                           font=dict(color="white", size=12),
                           bgcolor=color, opacity=0.9,
                           row=1, col=1)

        # annotate end if exists
        if mark.get("end") is not None:
            fig.add_annotation(x=end, y=y_val,
                               text=f"{mark['label']} (end)",
                               showarrow=True, arrowhead=2, ax=0, ay=-30,
                               font=dict(color="white", size=11),
                               bgcolor=color, opacity=0.9,
                               row=1, col=1)

    # Table on right
    table_cols = ["id","index","gas_resistance","status","gas_index","label"]
    present_cols = [c for c in table_cols if c in d.columns or c == "label"]
    # for table, ensure 'label' column exists in d (it does not exist as a DataFrame column — it's in buffer rows),
    # but we can add it safely to d for display:
    if "label" not in d.columns:
        d = d.copy()
        d["label"] = [r.get("label","") for r in d.to_dict('records')]

    d_sorted = d.sort_values("millis", ascending=False) if "millis" in d.columns else d
    fig.add_trace(go.Table(header=dict(values=[f"<b>{c}</b>" for c in present_cols], fill_color="#111", font=dict(color="white")),
                           cells=dict(values=[d_sorted[c] for c in present_cols], fill_color="#1f2937", font=dict(color="white"))),
                  row=1, col=2)

    fig.update_layout(template="plotly_dark", hovermode="x unified",
                      height=1100, width=1500,
                      title=f"Sensor: {selected_sensor} | Recording Label: {current_label if current_label else 'None'}")
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=3, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=4, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=4, col=1, rangeslider_visible=True, rangeslider_thickness=0.1)
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
    if n and value:
        set_label(value)
    style = {"color":"white","marginTop":"10px","fontSize":"22px","fontWeight":"bold","padding":"10px","borderRadius":"8px","textAlign":"center",
             "backgroundColor": _label_color(current_label)}
    return f"Current Label: {current_label if current_label else 'None'}", style

@app.callback([Output("sensor-dropdown","options"), Output("sensor-dropdown","value")],
              Input("interval-refresh","n_intervals"), State("sensor-dropdown","value"))
def update_sensors(n,current_value):
    df = load_data()
    if df.empty or "sensor_key" not in df.columns:
        return [], None
    sensors = df["sensor_key"].unique()
    value = current_value if current_value in sensors else sensors[0]
    return [{"label": s, "value": s} for s in sensors], value

@app.callback(Output("live-graph","figure"), [Input("interval-refresh","n_intervals"), Input("sensor-dropdown","value")])
def update_graph(n, selected_sensor):
    df = load_data()
    fig, _ = make_figure(df, selected_sensor)
    return fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader, daemon=True).start()
    app.run(debug=True, use_reloader=False)
