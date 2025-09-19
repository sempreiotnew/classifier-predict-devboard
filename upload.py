import pandas as pd
import base64, io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

# ---------------- LOAD DATA ----------------
def load_data(contents):
    if contents is None:
        return pd.DataFrame()
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        df = pd.read_csv(io.StringIO(decoded.decode("latin-1")))

    numeric_cols = ["millis","gas_resistance","temperature","pressure","humidity","index","gas_index"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "id" in df.columns and "index" in df.columns:
        df["sensor_key"] = df["id"].astype(str) + "_S" + df["index"].astype(str)
    return df

# ---------------- MAKE FIGURE ----------------
def make_figure(df, selected_sensor):
    needed_cols = ["millis","gas_resistance","temperature","pressure","humidity","id","index","status","sensor_key","gas_index"]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype=float if c in ["millis","gas_resistance","temperature","pressure","humidity","index","gas_index"] else object)

    sensors = df["sensor_key"].unique() if "sensor_key" in df.columns else []
    d = df[df["sensor_key"] == selected_sensor].sort_values("millis") if selected_sensor in sensors else pd.DataFrame()

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"type":"scatter"},{"type":"table"}],
               [{"type":"scatter"},{"type":"table"}],
               [{"type":"scatter"}, None],
               [{"type":"scatter"}, None]],
        column_widths=[0.65,0.35], row_heights=[0.5,0.17,0.17,0.16],
        shared_xaxes=True, vertical_spacing=0.05
    )

    # --- Gas Resistance Curves ---
    if not d.empty:
        color_map = {100: "orange", 99: "cyan"}
        parallel_colors = ["red","green","blue","purple","magenta","yellow","lime","teal","pink","brown"]
        for gas_idx in sorted(d["gas_index"].dropna().unique()):
            gi = int(gas_idx)
            sub = d[d["gas_index"].astype(int) == gi]
            if gi in (99, 100):
                fig.add_trace(go.Scatter(x=sub["millis"], y=sub["gas_resistance"], mode="lines+markers",
                                         name=f"Forced {gi}", line=dict(color=color_map.get(gi,"gray"))),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=sub["millis"], y=sub["gas_resistance"], mode="lines+markers",
                                         name=f"Step {gi}", line=dict(color=parallel_colors[gi % len(parallel_colors)])),
                              row=1, col=1)

    # --- Temperature, Pressure, Humidity ---
    fig.add_trace(go.Scatter(x=d["millis"], y=d["temperature"], mode="lines+markers", name="Temperature (°C)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["millis"], y=d["pressure"], mode="lines+markers", name="Pressure (Pa)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=d["millis"], y=d["humidity"], mode="lines+markers", name="Humidity (%)"), row=4, col=1)

    # --- Tables ---
    table_cols = ["id","index","gas_resistance","status","gas_index"]
    present_cols = [c for c in table_cols if c in d.columns]
    d_sorted = d.sort_values("millis", ascending=False)
    fig.add_trace(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in present_cols], fill_color="#111", font=dict(color="white")),
        cells=dict(values=[d_sorted[c] for c in present_cols], fill_color="#1f2937", font=dict(color="white"))
    ), row=1, col=2)

    min_v = d["gas_resistance"].min() if not d.empty else 0
    max_v = d["gas_resistance"].max() if not d.empty else 0
    mean_v = d["gas_resistance"].mean() if not d.empty else 0
    fig.add_trace(go.Table(
        header=dict(values=["MIN","MAX","AVG"], fill_color="#111", font=dict(color="white")),
        cells=dict(values=[[f"{min_v:,.2f}"],[f"{max_v:,.2f}"],[f"{mean_v:,.2f}"]],
                   fill_color="#222222", font=dict(color="white"))
    ), row=2, col=2)

    # Layout
    fig.update_layout(template="plotly_dark", title=f"BME688 Dashboard - {selected_sensor}", hovermode="x unified", height=1150, width=1500)
    fig.update_yaxes(title_text="Gas Resistance (Ω, log scale)", row=1, col=1, type="log")
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=3, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=4, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=4, col=1, rangeslider_visible=True, rangeslider_thickness=0.10)

    return fig, sensors

# ------------------- DASH APP -------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 CSV Dashboard", style={"color":"white","textAlign":"center"}),
    dcc.Upload(
        id="upload-data",
        children=html.Div(["Drag and Drop or ", html.A("Select CSV File")]),
        style={"width":"50%","margin":"auto","padding":"20px","borderWidth":"2px","borderStyle":"dashed","borderRadius":"10px","textAlign":"center","color":"white"},
        multiple=False
    ),
    dcc.Dropdown(id="sensor-dropdown", options=[], value=None, style={"width":"400px","margin":"20px auto"}),
    dcc.Graph(id="graph")
], style={"backgroundColor":"#111","padding":"20px"})

# ------------------- CALLBACKS -------------------
@app.callback(
    [Output("sensor-dropdown","options"), Output("sensor-dropdown","value"), Output("graph","figure")],
    [Input("upload-data","contents"), Input("sensor-dropdown","value")]
)
def update_dashboard(contents, selected_sensor):
    df = load_data(contents)
    if df.empty:
        return [], None, go.Figure()
    sensors = df["sensor_key"].unique() if "sensor_key" in df.columns else []
    if not sensors.any():
        return [], None, go.Figure()
    if selected_sensor not in sensors:
        selected_sensor = sensors[0]
    options = [{"label":s,"value":s} for s in sensors]
    fig, _ = make_figure(df, selected_sensor)
    return options, selected_sensor, fig

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)