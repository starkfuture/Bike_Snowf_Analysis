import subprocess
import sys

#-------------------------------------------------------------------
# IMPLEMENTED BY: MMM + ChatGPT (performance pass)
# DATE: 22/08/2025
# NOTES:
#   - WebGL rendering (ScatterGL) for large series.
#   - Server-side windowing & downsampling to cap points per query.
#   - Optional plotly-resampler integration (if available).
#   - Basic Flask-Caching to memoize Snowflake reads per VIN/ride/range.
#   - Lighter legend by default-hiding very dense groups (Cells).
#-------------------------------------------------------------------
HIDE_BUTTONS = {
    'DC Link', 'DC Bus', 'Cable Damaged', 'Battery V-', 'Battery V+',
    'throttle_not_closed_on_engage', 'throttle_not_closed_on_turning_on', 'DC Bus Δ',
    # 👇 nuevos: ocultamos los flags individuales
    'Flag: Cable Damaged', 'Flag: Throttle not closed on engage', 'Flag: Throttle not closed on turning on',
    'Flag: DC Bus Δ>80V', 'Flag: Fault Active'
}


# Ensure required packages are installed (best-effort)
required_packages = [
    "dash",
    "dash-bootstrap-components",
    "pandas",
    "plotly",
    "snowflake-connector-python",
    "flask-caching",
    "plotly-resampler"
]
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception:
            pass  # allow app to continue even if optional deps fail

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import ast
import plotly.graph_objects as go
import plotly.express as px
import snowflake.connector
from flask_caching import Cache

# Optional plotly-resampler
try:
    from plotly_resampler import FigureResampler, register_plotly_resampler
    register_plotly_resampler(mode="auto")
    HAS_PRS = True
except Exception:
    HAS_PRS = False

# -------------------- App & cache --------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

cache = Cache(server, config={
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300  # seconds
})

user_sessions = {}

# -------------------- Config --------------------
# Cap total plotted points to keep UI responsive (per figure)
MAX_POINTS_TOTAL = 120_000
# Default visible window (minutes) when opening a ride
DEFAULT_WINDOW_MIN = 10.0
# When many traces (e.g., rcells), hide them by default
HIDE_CELL_TRACES_BY_DEFAULT = False

# -------------------- Helpers --------------------
HUMIDITY_VARS = [
    "neg_battery_humidity",
    "pos_battery_humidity",
    "vcu_humidity",
    "inverter_humidity",
]

def is_humidity(v: str) -> bool:
    return v in HUMIDITY_VARS or v.endswith("_humidity")

def connect_to_snowflake(user_email):
    try:
        return snowflake.connector.connect(
            account="jv48965.eu-central-1",
            authenticator="externalbrowser",
            user=user_email,
            warehouse="TESTING_WH",
            database="ANALYTICS",
            schema="BIKE"
        )
    except Exception as e:
        print("Error:", repr(e))

def get_columns_for_vin(vin):
    fw_vars = [
        'pre_charge_neg_version','pre_charge_pos_version','esp_bottom_version','esp_top_version',
        'vcu_pic_version','dock_version','app_version_number','fw_fs_version',
        'map_switch_version','gate_version',
    ]
    humidity_vars = [
        'neg_battery_humidity','pos_battery_humidity','vcu_humidity','inverter_humidity'
    ]
    always = ['ride_time_ms','fault_bits','fault_bits_raw']

    # ✅ todas las temperaturas (battery array + scalars + VCU + inverter + gate)
    temps = [
        'battery_temps',
        'neg_battery_temp','pos_battery_temp','battery_temp_min','battery_temp_max',
        'vcu_temp',
        'igbt_temperature',
        'inverter_igbt_temp1','inverter_igbt_temp2','inverter_igbt_temp3',
        'inverter_gate_pcb_ntcs1','inverter_gate_pcb_ntcs2','inverter_gate_pcb_ntcs3'
    ]
    other_vars = [
        'battery_charge_state_percentaje', 'throttle_position', 'speed_kmh',
        'motor_temperature', 'dc_link_inv', 'dc_bus_battery', 'throttle_cable_damaged', 'power_mode_activated_index',
        'iq', 'id', 'id_fb', 'iq_fb', 'id_ref', 'iq_ref', 'dc_bus_negative_volts', 'dc_bus_positive_volts',
        # 👇 AÑADIR ESTAS DOS FLAGS
        'throttle_not_closed_on_engage', 'throttle_not_closed_on_turning_on',
        'charger_connected', 'charger_plugged', 'charging_current',
        'gyro_x', 'gyro_y', 'gyro_z', 'pump_current', 'battery_current','motor_rpm',
    ]

    cells = ['cell_a_idx','cell_a_voltage','cell_b_idx','cell_b_voltage',
             'cell_c_idx','cell_c_voltage','cell_d_idx','cell_d_voltage','cells']
    accel_vars = ['accel_x','accel_y','accel_z']

    return always + cells + temps + other_vars + accel_vars + fw_vars + humidity_vars


@cache.memoize()
def get_ride_starts(email, vin):
    conn = user_sessions[email]["conn"]
    query = f"""
        SELECT ride_start_on_datetime, is_charging_ride, riding_time_mins
        FROM analytics.bike.varg_telemetry_agg
        WHERE vin = '{vin}'
        ORDER BY ride_start_on_datetime DESC
    """
    df = pd.read_sql(query, conn)
    df.columns = df.columns.str.lower()
    return df


@cache.memoize()
def get_min_max_ms(email, vin, ride_time):
    conn = user_sessions[email]["conn"]
    q = f"""
        SELECT MIN(ride_time_ms) AS mn, MAX(ride_time_ms) AS mx
        FROM analytics.bike.varg_telemetry
        WHERE vin='{vin}' AND ride_start_on_datetime='{ride_time}'
    """
    df = pd.read_sql(q, conn)
    return int(df.iloc[0,0] or 0), int(df.iloc[0,1] or 0)


def compute_step_for_downsample(n_rows, n_vars):
    # crude step to bound total points
    if n_rows <= 0 or n_vars <= 0:
        return 1
    target_per_series = max(200, int(MAX_POINTS_TOTAL / max(1, n_vars)))
    step = int(np.ceil(n_rows / target_per_series))
    return max(1, step)


@cache.memoize()
def get_telemetry_window(email, vin, ride_time, columns, start_ms, end_ms, step):
    conn = user_sessions[email]["conn"]
    # Server-side downsampling using a row_number modulus filter
    query = f"""
        WITH base AS (
            SELECT {', '.join(columns)},
                   ROW_NUMBER() OVER (ORDER BY ride_time_ms) AS rn
            FROM analytics.bike.varg_telemetry
            WHERE vin = '{vin}'
              AND ride_start_on_datetime = '{ride_time}'
              AND ride_time_ms BETWEEN {start_ms} AND {end_ms}
        )
        SELECT *
        FROM base
        WHERE MOD(rn, {step}) = 1
        ORDER BY ride_time_ms ASC
    """
    df = pd.read_sql(query, conn)
    df.columns = df.columns.str.lower()
    return df


# Map of fault bits to names
FAULT_BIT_MAP = {
    1:"FAULT_OVERVOLTAGE", 2:"FAULT_IOC", 4:"FAULT_ROTPOS_PARITY_POS", 8:"FAULT_ROTPOS_PARITY_AGC",
    16:"FAULT_ROTPOS_MAGL", 32:"FAULT_ROTPOS_MAGH", 64:"FAULT_ROTPOS_CORDIC", 128:"FAULT_12V_UNDERVOLT",
    256:"FAULT_IGBT_DESAT", 512:"FAULT_IGBT_TEMP", 1024:"FAULT_MOTOR_TEMP", 2048:"FAULT_MAXMOTOR_TEMP",
    8192:"FAULT_INVERTER_STOPPING", 16384:"FAULT_INVERTER_STARTING", 32768:"FAULT_SW_RESET"
}

# -------------------- Layout --------------------
app.layout = dbc.Container([
    html.H1("🔋 Bike Analysis Tool — Fast"),

    dbc.Row([
        dbc.Col(dbc.Input(id="email-input", placeholder="Enter Snowflake email", type="email")),
        dbc.Col(dbc.Button("Connect", id="connect-button", color="primary"))
    ], id="email-row", className="mb-3"),
    html.Div(id="connection-status"),

    #html.Div(id="ride-type-warning"),

    dbc.Row(dbc.Col(dbc.Input(id="vin-input", placeholder="Enter VIN", type="text", style={"display":"none"})), className="mb-3"),

    dcc.Dropdown(id="rides-dropdown", placeholder="Select ride start time", options=[], style={"display":"none"}),

    # Always create the RangeSlider (avoid nonexistent object errors)
    html.Div(id="range-slider-container", children=[
        dcc.RangeSlider(
            id="time-range-slider",
            min=0, max=1, step=0.1, value=[0,1],
            allowCross=False,
            tooltip={"placement":"bottom", "always_visible": False}
        )
    ], style={"display":"none", "marginBottom":"10px"}),

    html.Div(id="var-selection-container", style={"display":"none"}, children=[
        dbc.Checklist(options=[], value=[], id="variable-selection", inline=True)
    ]),

    dcc.Graph(id="telemetry-graph", style={"display":"none"}),

    html.Div(id="high-temp-list", style={"marginTop":"15px"}),
    html.Div(id="cells-disconnected-list", style={"marginTop":"15px"}),
    html.Div(id="cells-discharged-list", style={"marginTop":"15px"}),
    html.Div(id="fault-bits-display"),
    html.Div(id="fw-version-table"),

    html.Div([
        html.H5("Delta Measurement Tool"),
        html.Div(id="delta-output"),
        dbc.Button("Clear Selection", id="clear-selection", color="warning"),
        dcc.Store(id="clicked-points", data=[])
    ], className="mt-4"),
], fluid=True)

# -------------------- Callbacks --------------------
@app.callback(
    Output("connection-status", "children"),
    Output("vin-input", "style"),
    Output("email-row", "style"),
    Input("connect-button", "n_clicks"),
    State("email-input", "value")
)
def connect_cb(n_clicks, email):
    if n_clicks and email:
        try:
            conn = connect_to_snowflake(email)
            user_sessions[email] = {"conn": conn}
            return dbc.Alert(f"Connected as {email}", color="success"), {"display":"block"}, {"display":"none"}
        except Exception as e:
            return dbc.Alert(f"Connection failed: {e}", color="danger"), {"display":"none"}, {"display":"block"}
    return "", {"display":"none"}, {"display":"block"}


@app.callback(
    Output("rides-dropdown", "options"),
    Output("rides-dropdown", "style"),
    Input("vin-input", "value"),
    State("email-input", "value")
)
def load_rides(vin, email):
    if vin and email in user_sessions:
        df = get_ride_starts(email, vin)
        if df.empty:
            return [], {"display":"none"}
        def fmt_minutes(x):
            try:
                return f"{float(x):.1f} min"
            except Exception:
                return None
        options = []
        for _, row in df.iterrows():
            ride_dt = row['ride_start_on_datetime']
            is_ch = row.get('is_charging_ride')
            mins  = row.get('riding_time_mins')
            mins_s = fmt_minutes(mins)
            tag = 'Charging Ride ⚡' if bool(str(is_ch).lower() == 'true' or is_ch == 1) else 'Ride Session 🏍️'
            label = f"{ride_dt} — {tag}"
            if mins_s:
                label += f" — {mins_s}"
            options.append({"label": label, "value": str(ride_dt)})
        return options, {"display":"block"}
    return [], {"display":"none"}


@app.callback(
    Output("range-slider-container", "children"),
    Output("range-slider-container", "style"),
    Input("rides-dropdown", "value"),
    State("email-input", "value"),
    State("vin-input", "value")
)
def build_slider(ride_time, email, vin):
    if not (ride_time and email in user_sessions and vin):
        return None, {"display":"none"}
    mn, mx = get_min_max_ms(email, vin, ride_time)
    if mn >= mx:
        return None, {"display":"none"}
    mn_min = mn / 60000.0
    mx_min = mx / 60000.0
    # Default window is last DEFAULT_WINDOW_MIN minutes
    start_default = max(mn_min, mx_min - DEFAULT_WINDOW_MIN)
    slider = dcc.RangeSlider(
        id="time-range-slider",
        min=mn_min, max=mx_min, step=0.001,
        value=[start_default, mx_min],
        allowCross=False,
        tooltip={"placement":"bottom", "always_visible": False}
    )
    return html.Div([
        html.Div("Time Window (min)"),
        slider
    ]), {"display":"block"}


@app.callback(
    Output("var-selection-container", "style"),
    Output("variable-selection", "options"),
    Output("variable-selection", "value"),
    Output("telemetry-graph", "style"),
    Output("telemetry-graph", "figure"),
    Output("high-temp-list", "children"),
 #   Output("ride-type-warning", "children"),
    Output("fault-bits-display", "children"),
    Output("fw-version-table", "children"),
    Output("cells-disconnected-list", "children"),
    Output("cells-discharged-list", "children"),
    Input("rides-dropdown", "value"),
    Input("variable-selection", "value"),
    Input("time-range-slider", "value"),
    State("vin-input", "value"),
    State("email-input", "value")
)
def update_graph(ride_time, selected_vars, slider_range, vin, email):
    # Guard
    if not (ride_time and vin and email in user_sessions and slider_range):
        empty = {"display":"none"}
        return empty, [], [], empty, {}, None, None, "", "", ""

    cols = get_columns_for_vin(vin)

    # Window selection
    mn_ms, mx_ms = get_min_max_ms(email, vin, ride_time)
    start_ms = max(mn_ms, int(float(slider_range[0]) * 60000))
    end_ms = min(mx_ms, int(float(slider_range[1]) * 60000) + 500)

    # Pequeño colchón para no perdernos el último sample por redondeos
    end_ms += 500  # +0.5s

    # Estimate downsample step from row count (approx via min/max only)
    total_range_ms = max(1, mx_ms - mn_ms)
    window_frac = min(1.0, max(0.0, (end_ms - start_ms) / total_range_ms))

    # crude n_rows guess via agg table (fallback to cap)
    # We'll fetch without step first rowcount estimate by a light query could be added;
    # use a conservative per-var cap
    n_vars_guess = 20  # rough guess; adjusted after building df
    n_rows_guess = int(200_000 * window_frac)  # heuristic
    step = compute_step_for_downsample(n_rows_guess, n_vars_guess)

    # tras: step = compute_step_for_downsample(...)
    wants_cells = any(v == 'Cells' or (isinstance(v, str) and v.startswith('Cell_'))
                      for v in (selected_vars or []))
    if wants_cells:
        step = max(1, step // 4)  # más denso para no perder picos ~0 V

    telemetry_df = get_telemetry_window(email, vin, ride_time, cols, start_ms, end_ms, step)

    if telemetry_df.empty:
        empty = {"display":"block"}
        return empty, [], [], {"display":"none"}, {}, html.Div("No data in window"), dbc.Alert("No data", color="warning"), html.Div(), html.Div(), html.Div()

    telemetry_df = telemetry_df.where(~telemetry_df.isnull(), 'N/A')
    telemetry_df['ride_time_min'] = (telemetry_df['ride_time_ms'] / 60000.0)

    # --------- Build long dataframe (vectorized where possible) ---------
    expanded_rows = []

    # Accel
    for ax in ('accel_x','accel_y','accel_z'):
        if ax not in telemetry_df.columns:
            break
    else:
        # All present
        ax_g = telemetry_df['accel_x'].astype(float) / 2048.0
        ay_g = telemetry_df['accel_y'].astype(float) / 2048.0
        az_g = telemetry_df['accel_z'].astype(float) / 2048.0
        total_g = (ax_g**2 + ay_g**2 + az_g**2) ** 0.5
        tmp = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Accel X (g)': ax_g,
            'Accel Y (g)': ay_g,
            'Accel Z (g)': az_g,
        })
        expanded_rows.append(
            tmp.melt(id_vars='ride_time_min', var_name='Variable', value_name='Value')
        )

    # Cells from array first
    use_cells_array = (
        'cells' in telemetry_df.columns and
        telemetry_df['cells'].apply(lambda x: isinstance(x, str) and x not in ['N/A', '', '[]']).any()
    )
    if use_cells_array:
        # Parse safely; ignore errors
        parsed = telemetry_df['cells'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x not in ['N/A','','[]'] else [])
        # Scale to volts
        parsed = parsed.apply(lambda arr: [v/1000.0 for v in arr])
        # Explode
        tmp = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'cells_list': parsed
        }).explode('cells_list')
        tmp.dropna(subset=['cells_list'], inplace=True)
        tmp['Variable'] = 'Cell_'
        # Enumerate index within each timestamp
        tmp['idx'] = tmp.groupby(level=0).cumcount() + 1
        tmp['Variable'] = tmp['Variable'] + tmp['idx'].astype(str)
        tmp.rename(columns={'cells_list':'Value'}, inplace=True)
        tmp = tmp[['ride_time_min','Variable','Value']]
        expanded_rows.append(tmp)
    else:
        # Fallback individual cell_* columns
        cell_cols = [
            ('cell_a_idx','cell_a_voltage'),('cell_b_idx','cell_b_voltage'),
            ('cell_c_idx','cell_c_voltage'),('cell_d_idx','cell_d_voltage')
        ]
        long_list = []
        for idx_col, val_col in cell_cols:
            if idx_col in telemetry_df.columns and val_col in telemetry_df.columns:
                valid = telemetry_df[[idx_col,val_col,'ride_time_min']].copy()
                valid = valid.dropna()
                try:
                    valid[idx_col] = valid[idx_col].astype(int)
                    valid[val_col] = valid[val_col].astype(float)
                except Exception:
                    continue
                valid['Variable'] = valid[idx_col].apply(lambda i: f"Cell_{i}")
                valid.rename(columns={val_col:'Value'}, inplace=True)
                long_list.append(valid[['ride_time_min','Variable','Value']])
        if long_list:
            expanded_rows.append(pd.concat(long_list, ignore_index=True))

    # Battery temp array
    if 'battery_temps' in telemetry_df.columns:
        parsed = telemetry_df['battery_temps'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x not in ['N/A','','[]'] else [])
        parsed = parsed.apply(lambda arr: [v/10.0 for v in arr])
        tmp = pd.DataFrame({'ride_time_min': telemetry_df['ride_time_min'], 'btemps': parsed}).explode('btemps')
        tmp.dropna(subset=['btemps'], inplace=True)
        tmp['Variable'] = 'Batt Temp '
        tmp['idx'] = tmp.groupby(level=0).cumcount() + 1
        tmp['Variable'] = tmp['Variable'] + tmp['idx'].astype(str)
        tmp.rename(columns={'btemps':'Value'}, inplace=True)
        tmp = tmp[['ride_time_min','Variable','Value']]
        expanded_rows.append(tmp)

    # Battery temp scalars
    batt_temp_scalar_map = [
        ('neg_battery_temp','Batt Temp PCH Neg'), ('pos_battery_temp','Batt Temp PCH Pos'),
        ('battery_temp_min','Batt Temp Min'), ('battery_temp_max','Batt Temp Max')
    ]
    for raw, name in batt_temp_scalar_map:
        if raw in telemetry_df.columns:
            s = pd.to_numeric(telemetry_df[raw], errors='coerce').dropna()
            if not s.empty:
                tmp = pd.DataFrame({'ride_time_min': telemetry_df.loc[s.index, 'ride_time_min'], 'Variable': name, 'Value': s})
                expanded_rows.append(tmp)

    # Inverter temps
    inverter_temp_map = [
        ('igbt_temperature','Inv IGBT Temp'), ('inverter_igbt_temp1','Inv IGBT Temp1'), ('inverter_igbt_temp2','Inv IGBT Temp2'),
        ('inverter_igbt_temp3','Inv IGBT Temp3'), ('inverter_gate_pcb_ntcs1','Gate PCB NTC1'),
        ('inverter_gate_pcb_ntcs2','Gate PCB NTC2'), ('inverter_gate_pcb_ntcs3','Gate PCB NTC3')
    ]
    for raw, name in inverter_temp_map:
        if raw in telemetry_df.columns:
            s = pd.to_numeric(telemetry_df[raw], errors='coerce').dropna()
            if not s.empty:
                tmp = pd.DataFrame({'ride_time_min': telemetry_df.loc[s.index, 'ride_time_min'], 'Variable': name, 'Value': s})
                expanded_rows.append(tmp)

    # Others / voltages
    mappings = [
        ('battery_charge_state_percentaje', 'Battery %'),
        ('throttle_position', 'Throttle'),
        ('speed_kmh', 'Speed'),
        ('motor_temperature', 'Motor Temp'),
        ('dc_link_inv', 'DC Link'),
        ('dc_bus_battery', 'DC Bus'),
        ('throttle_cable_damaged', 'Cable Damaged'),
        ('dc_bus_negative_volts', 'Battery V-'),  # 👈 agregado
        ('dc_bus_positive_volts', 'Battery V+'),  # 👈 agregado
        ('inverter_status', 'Inverter'),
        ('power_mode_activated_index', 'Power Mode Activated Index'),
        # --- nuevas variables ---
        ('iq', 'iq'),
        ('id', 'id'),
        ('id_fb', 'id_fb'),
        ('iq_fb', 'iq_fb'),
        ('id_ref', 'id_ref'),
        ('iq_ref', 'iq_ref'),
        ('gyro_x', 'gyro_x'),
        ('gyro_y', 'gyro_y'),
        ('gyro_z', 'gyro_z'),
        ('pump_current', 'pump_current'),
        ('battery_current', 'battery_current'),
        ('motor_rpm', 'motor_rpm'),

        # 👉 VCU + Inverter + Gate temps
        ('vcu_temp', 'VCU Temp'),
        ('igbt_temperature', 'Inv IGBT Temp'),
        ('inverter_igbt_temp1', 'Inv IGBT Temp1'),
        ('inverter_igbt_temp2', 'Inv IGBT Temp2'),
        ('inverter_igbt_temp3', 'Inv IGBT Temp3'),
        ('inverter_gate_pcb_ntcs1', 'Gate PCB NTC1'),
        ('inverter_gate_pcb_ntcs2', 'Gate PCB NTC2'),
        ('inverter_gate_pcb_ntcs3', 'Gate PCB NTC3'),

        # 👉 Acceleration (si prefieres en g: conviértelo más abajo)
        ('accel_x', 'Accel X'),
        ('accel_y', 'Accel Y'),
        ('accel_z', 'Accel Z'),

        # 👉 Humidity (se verán en % si vienen ya así; si llegan 0–1, puedes multiplicar por 100 antes)
        ('neg_battery_humidity', 'Humidity Battery − (neg)'),
        ('pos_battery_humidity', 'Humidity Battery + (pos)'),
        ('vcu_humidity', 'Humidity VCU'),
        ('inverter_humidity', 'Humidity Inverter PCB'),

        # 👉 Battery temp scalars
        ('neg_battery_temp', 'Batt Temp PCH Neg'),
        ('pos_battery_temp', 'Batt Temp PCH Pos'),
        ('battery_temp_min', 'Batt Temp Min'),
        ('battery_temp_max', 'Batt Temp Max'),

        ('charger_connected', 'Charger Connected'),
        ('charger_plugged', 'Charger Plugged'),
        ('charging_current', 'Charging Current (A)')
    ]

    # --------- Flags as normal 0/1 series (replacing overlays) ---------
    def to01(series):
        s_str = series.astype(str).str.strip().str.lower()
        s_num = pd.to_numeric(series, errors="coerce")
        return ((s_num == 1) | s_str.isin(["1", "true", "t", "yes", "y", "on"])).astype(int)

    # Cable Damaged (si existe en DF base o ya mapeado)
    if 'throttle_cable_damaged' in telemetry_df.columns:
        f = to01(telemetry_df['throttle_cable_damaged'])
        tmp = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': 'Flag: Cable Damaged',
            'Value': f
        })
        expanded_rows.append(tmp)

    # Throttle flags (si existen)
    if 'throttle_not_closed_on_engage' in telemetry_df.columns:
        f = to01(telemetry_df['throttle_not_closed_on_engage'])
        tmp = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': 'Flag: Throttle not closed on engage',
            'Value': f
        })
        expanded_rows.append(tmp)

    if 'throttle_not_closed_on_turning_on' in telemetry_df.columns:
        f = to01(telemetry_df['throttle_not_closed_on_turning_on'])
        tmp = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': 'Flag: Throttle not closed on turning on',
            'Value': f
        })
        expanded_rows.append(tmp)

    # DC Bus Δ > 80V → flag 0/1
    if {'dc_bus_positive_volts', 'dc_bus_negative_volts'}.issubset(telemetry_df.columns):
        pos = pd.to_numeric(telemetry_df['dc_bus_positive_volts'], errors='coerce')
        neg = pd.to_numeric(telemetry_df['dc_bus_negative_volts'], errors='coerce')
        diff = (pos - neg).abs()
        flag = (diff > 80).astype(int)
        tmp = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': 'Flag: DC Bus Δ>80V',
            'Value': flag
        }).dropna(subset=['Value'])
        expanded_rows.append(tmp)

    # --- Escala Battery V+ / V- por 10 antes de cálculos y gráficos ---
    if {'dc_bus_positive_volts', 'dc_bus_negative_volts'}.issubset(telemetry_df.columns):
        telemetry_df['dc_bus_positive_volts'] = pd.to_numeric(
            telemetry_df['dc_bus_positive_volts'], errors='coerce'
        )
        telemetry_df['dc_bus_negative_volts'] = pd.to_numeric(
            telemetry_df['dc_bus_negative_volts'], errors='coerce'
        )

    # --- Series por cada fault bit (0/1) ---
    # Combinamos fault_bits y fault_bits_raw con OR vectorizado
    fb_series = pd.to_numeric(telemetry_df.get('fault_bits'), errors='coerce').fillna(0).astype(int)
    fbr_series = pd.to_numeric(telemetry_df.get('fault_bits_raw'), errors='coerce').fillna(0).astype(int)
    fault_combined = (fb_series | fbr_series)

    # Una curva 0/1 por cada fault definido
    for bit, name in FAULT_BIT_MAP.items():
        vals = ((fault_combined & bit) != 0).astype(int)
        if vals.any():
            expanded_rows.append(pd.DataFrame({
                'ride_time_min': telemetry_df['ride_time_min'],
                'Variable': f'Flag: {name}',  # Mantén "Flag:" para que solo salgan al elegir "Flags"
                'Value': vals
            }))

    # --- Charger group (Connected/Plugged como 0/1; Current como numérico) ---
    if 'charger_connected' in telemetry_df.columns:
        f = to01(telemetry_df['charger_connected'])
        expanded_rows.append(pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': 'Charger Connected',
            'Value': f
        }))

    if 'charger_plugged' in telemetry_df.columns:
        f = to01(telemetry_df['charger_plugged'])
        expanded_rows.append(pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': 'Charger Plugged',
            'Value': f
        }))

    if 'charging_current' in telemetry_df.columns:
        s = pd.to_numeric(telemetry_df['charging_current'], errors='coerce')
        if s.notna().any():
            expanded_rows.append(pd.DataFrame({
                'ride_time_min': telemetry_df['ride_time_min'],
                'Variable': 'Charging Current (A)',
                'Value': s
            }))

    present_hums = [c for c in HUMIDITY_VARS if c in telemetry_df.columns]
    if present_hums:
        hum_df = telemetry_df[['ride_time_min'] + present_hums].copy()
        # Si llegan en 0–1, escala a %
        for col in present_hums:
            s = pd.to_numeric(hum_df[col], errors='coerce')
            if s.notna().any():
                mx = s.max()
                if mx is not None and mx <= 1.5:
                    hum_df[col] = s * 100.0
                else:
                    hum_df[col] = s
        # Nombres bonitos
        PRETTY_HUM = {
            "neg_battery_humidity": "Humidity Battery − (neg)",
            "pos_battery_humidity": "Humidity Battery + (pos)",
            "vcu_humidity": "Humidity VCU",
            "inverter_humidity": "Humidity Inverter PCB",
        }
        hum_df = hum_df.melt(id_vars='ride_time_min', var_name='raw', value_name='Value')
        hum_df['Variable'] = hum_df['raw'].map(PRETTY_HUM).fillna(hum_df['raw'].str.replace('_', ' ').str.title())
        hum_df.drop(columns=['raw'], inplace=True)
        expanded_rows.append(hum_df)

    for raw, name in mappings:
        if raw in telemetry_df.columns:
            s = pd.to_numeric(telemetry_df[raw], errors='coerce').dropna()
            if not s.empty:
                tmp = pd.DataFrame({'ride_time_min': telemetry_df.loc[s.index, 'ride_time_min'], 'Variable': name, 'Value': s})
                expanded_rows.append(tmp)

    if not expanded_rows:
        empty = {"display":"block"}
        return empty, [], [], {"display":"none"}, {}, html.Div("No decodable data"), dbc.Alert("No data", color="warning"), html.Div(), html.Div(), html.Div()

    df_expanded = pd.concat(expanded_rows, ignore_index=True)

    # 🔧 Conversión segura: cualquier string/None → NaN
    if 'Value' in df_expanded.columns:
        df_expanded['Value'] = pd.to_numeric(df_expanded['Value'], errors='coerce')

    # Group classification for ordering
    def classify(v):
        vl = str(v).lower()
        if vl.startswith('flag:'):
            return '0_Flags'  # 👈 primero
        if str(v).startswith('Accel'):
            return '1_Acceleration'
        if str(v).startswith('Batt Temp'):
            return '2_BatteryTemps'
        if any(k in vl for k in ['inv igbt temp', 'gate pcb ntc']):
            return '3_InverterTemp'
        if 'humidity' in vl:
            return '3_5_Humidity'
        if str(v).startswith('Cell_') or any(k in vl for k in ['v+', 'v-', 'volt', 'dc bus', 'dc link']):
            return '4_Voltages'
        if 'charger' in vl:
            return '3_6_Charger'
        return '5_Others'

    df_expanded['Group'] = df_expanded['Variable'].apply(classify)

    all_vars = sorted(df_expanded['Variable'].unique())

    options = [
                  {'label': 'Cells', 'value': 'Cells'},
                  {'label': 'Temperatures', 'value': 'Temperatures'},
                  {'label': 'Acceleration', 'value': 'Acceleration'},
                  {'label': 'Humidity', 'value': 'Humidity'},
                  {'label': 'Voltages', 'value': 'Voltages'},
                  {'label': 'Flags', 'value': 'Flags'},
                  {'label': 'Charger', 'value': 'Charger'},# 👈 NUEVO
              ] + [
                  {'label': v, 'value': v}
                  for v in all_vars
                  if not (
                v.startswith('Cell_')
                or v.startswith('Temp_')
                or 'Temp' in v
                or 'NTC' in v
                or v == 'VCU Temp'
                or v.startswith('Accel ')
                or 'Humidity' in v
                or v.startswith('Flag:')
                or 'Charger' in v
        )
                     and v not in HIDE_BUTTONS
              ]

    sel = selected_vars or ['Flags']
    expanded_sel = []
    for v in sel:
        if v == 'Cells':
            expanded_sel += [x for x in all_vars if x.startswith('Cell_')]

        elif v == 'Temperatures':
            expanded_sel += [x for x in all_vars if (
                    x.startswith('Temp_')  # battery_temps array
                    or x.startswith('Batt Temp')  # scalars: neg/pos/min/max
                    or 'Inv IGBT Temp' in x  # inverter temps
                    or 'Gate PCB NTC' in x  # gate temps
                    or x == 'VCU Temp'  # vcu temp
            )]
        elif v == 'Voltages':
            expanded_sel += [x for x in all_vars if x in ['DC Link', 'DC Bus', 'Battery V-', 'Battery V+']]
        elif v == 'Acceleration':
            expanded_sel += [x for x in all_vars if x.startswith('Accel ')]
        elif v == 'Humidity':
            expanded_sel += [x for x in all_vars if 'Humidity' in x]
        elif v == 'Flags':  # 👈 NUEVO
            expanded_sel += [x for x in all_vars if x.startswith('Flag:')]
        elif v == 'Charger':
            expanded_sel += [x for x in all_vars if x in [
                'Charger Connected',
                'Charger Plugged',
                'Charging Current (A)'
            ]]

        else:
            expanded_sel.append(v)

    df_filtered = df_expanded[df_expanded['Variable'].isin(expanded_sel)]
    # Asegura orden por variable y tiempo para PRS
    df_filtered = df_filtered.sort_values(['Variable', 'ride_time_min']).reset_index(drop=True)

    # Order legend
    order_vars = (
        df_filtered[['Variable','Group']].drop_duplicates().sort_values(['Group','Variable'])['Variable'].tolist()
    )

    # ------------- Build figure -------------
    # Prefer FigureResampler if available for dynamic downsampling in-browser
    if HAS_PRS:
        fig_rs = FigureResampler(go.Figure())
        for var in order_vars:
            d = df_filtered[df_filtered['Variable'] == var]
            if d.empty:
                continue
            # Orden y limpieza para evitar el assert de PRS
            d = d.sort_values('ride_time_min').dropna(subset=['ride_time_min', 'Value'])
            if d.empty:
                continue
            # (Opcional) eliminar duplicados exactos de X si existieran
            d = d[~d['ride_time_min'].duplicated(keep='first')]

            vis = True
            if HIDE_CELL_TRACES_BY_DEFAULT and var.startswith('Cell_'):
                vis = 'legendonly'

            fig_rs.add_trace(
                go.Scattergl(name=var, mode='lines', visible=vis),
                hf_x=d['ride_time_min'].to_numpy(),
                hf_y=d['Value'].to_numpy(),
            )
        fig_rs.update_layout(
            xaxis_title='Tiempo (min)', legend_title='Variables',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig = fig_rs

    else:
        # Fallback: regular figure with WebGL + pre-downsampling from SQL
        fig = px.line(
            df_filtered, x='ride_time_min', y='Value', color='Variable',
            category_orders={'Variable': order_vars}, render_mode='webgl',
            markers=True  # hace visibles los puntos sueltos
        )
        if HIDE_CELL_TRACES_BY_DEFAULT:
            for tr in fig.data:
                if str(tr.name).startswith('Cell_'):
                    tr.visible = 'legendonly'
        fig.update_layout(
            xaxis_title='Tiempo (min)', legend_title='Variables',
            margin=dict(l=10, r=10, t=40, b=10)
        )

    # Firmware table
    fw_vars = [
        'pre_charge_neg_version', 'pre_charge_pos_version', 'esp_bottom_version', 'esp_top_version',
        'vcu_pic_version', 'dock_version', 'app_version_number', 'fw_fs_version',
        # Añadidas:
        'map_switch_version', 'gate_version'
    ]
    fw_rows = []
    for fw in fw_vars:
        if fw in telemetry_df.columns and not telemetry_df.empty:
            first = telemetry_df.iloc[0][fw]
            last = telemetry_df.iloc[-1][fw]
        else:
            first, last = 'N/A', 'N/A'
        fw_rows.append(html.Tr([html.Td(fw), html.Td(first), html.Td(last)]))
    fw_table = dbc.Table([
        html.Thead(html.Tr([html.Th("FW Var"), html.Th("First"), html.Th("Last")])) ,
        html.Tbody(fw_rows)
    ], bordered=True)

    # High battery temps (> 100C)
    high_batt = df_expanded[(df_expanded['Variable'].str.startswith('Batt Temp')) & (df_expanded['Value'] > 100)]
    if high_batt.empty:
        high_disp = html.Div([html.H4("Battery Temps above 100 °C"), html.Div("None")])
    else:
        items = [f"{r.Variable} = {r.Value:.1f} °C at {r.ride_time_min:.2f} min" for r in high_batt.itertuples()]
        high_disp = html.Div([html.H4("Battery Temps above 100 °C")] + [html.Div(x) for x in items])

    # Cells disconnected / discharged
    disc = {}
    disch = {}
    for var in sorted(v for v in all_vars if str(v).startswith('Cell_')):
        rows = df_expanded[df_expanded['Variable'] == var].sort_values('ride_time_min')
        if rows.empty:
            continue
        vals_d, segs_d, vals_c, segs_c = [], [], [], []
        start_d = start_c = None
        for r in rows.itertuples():
            t = r.ride_time_min; v = r.Value
            if v <= 0.001:
                vals_d.append(v)
                if start_d is None:
                    start_d = t
            else:
                if start_d is not None:
                    segs_d.append(t - start_d); start_d = None
            if 0.001 < v < 2.5:
                vals_c.append(v)
                if start_c is None:
                    start_c = t
            else:
                if start_c is not None:
                    segs_c.append(t - start_c); start_c = None
        if start_d is not None:
            segs_d.append(rows['ride_time_min'].iloc[-1] - start_d)
        if start_c is not None:
            segs_c.append(rows['ride_time_min'].iloc[-1] - start_c)
        if vals_d:
            disc[var] = (min(vals_d), max(segs_d) if segs_d else 0)
        if vals_c:
            disch[var] = (min(vals_c), max(segs_c) if segs_c else 0)

    def table_from_dict(dct, title, min_label):
        if not dct:
            body = [html.Tr([html.Td("None"), html.Td("—"), html.Td("—")])]
        else:
            body = [html.Tr([html.Td(k), html.Td(f"{v[0]:.3f} V"), html.Td(f"{v[1]*60_000:.0f} ms")]) for k,v in dct.items()]
        return html.Div([
            html.H4(title),
            dbc.Table([
                html.Thead(html.Tr([html.Th("Cell"), html.Th(min_label), html.Th("Max Duration")])) ,
                html.Tbody(body)
            ], bordered=True)
        ])

    disc_disp  = table_from_dict(disc,  "Cells Disconnected (≤ 1 mV)", "Min Voltage")
    disch_disp = table_from_dict(disch, "Cells Discharged (1 mV < V < 2.5 V)", "Min Voltage")

    faults_by_name = {name: [] for name in FAULT_BIT_MAP.values()}

    fb = telemetry_df.get('fault_bits')
    fbr = telemetry_df.get('fault_bits_raw')
    if fb is not None or fbr is not None:
        for _, row in telemetry_df.iterrows():
            val = 0
            # sumamos (OR) ambos campos si existen
            for key in ('fault_bits', 'fault_bits_raw'):
                try:
                    rv = row.get(key)
                except AttributeError:
                    rv = None
                try:
                    rv = int(rv) if rv not in (None, 'N/A', '') else 0
                except Exception:
                    rv = 0
                val |= rv

            if val:
                t = float(row['ride_time_min'])
                for bit, name in FAULT_BIT_MAP.items():
                    if val & bit:
                        faults_by_name[name].append(t)

    # 🔹 Añadir cada fault como una serie independiente (0/1)
    for fault_name, times in faults_by_name.items():
        if not times:
            continue
        # Creamos serie 0/1 en todo el rango de tiempo
        fault_series = pd.DataFrame({
            'ride_time_min': telemetry_df['ride_time_min'],
            'Variable': fault_name,
            'Value': telemetry_df['ride_time_min'].apply(lambda t: 1 if t in times else 0)
        })
        expanded_rows.append(fault_series)

    # Fault LEDs
    fault_leds = []
    for name in FAULT_BIT_MAP.values():
        times = faults_by_name.get(name, [])
        icon = '🔴' if times else '🟢'
        status = f"ON (first at {times[0]:.2f} min)" if times else 'OFF'
        fault_leds.append(
            html.Div([
                html.Span(icon, style={'marginRight': '5px'}),
                html.Span(name),
                html.Span(f' — {status}', style={'marginLeft': '5px'})
            ], style={'marginBottom': '5px'})
        )
    fault_display = html.Div([html.H4('Fault Indicator LEDs')] + fault_leds)

    return (
        {"display":"block"},  # selector visible
        options,
        sel,
        {"display":"block"},
        fig,
        high_disp,
        #ride_msg,
        fault_display,
        fw_table,
        disc_disp,
        disch_disp
    )


# Clear selection callback
@app.callback(
    Output("clicked-points", "data"),
    Input("telemetry-graph", "clickData"),
    Input("clear-selection", "n_clicks"),
    State("clicked-points", "data")
)
def clear_or_append(clickData, clear_clicks, stored):
    triggered = ctx.triggered_id
    if triggered == "clear-selection":
        return []
    if clickData:
        pts = stored or []
        pts.append(clickData["points"][0])
        if len(pts) > 2:
            pts = pts[-2:]
        return pts
    return stored


# Delta measurement callback
@app.callback(
    Output("delta-output", "children"),
    Input("clicked-points", "data")
)
def display_delta(clicked_points):
    if not clicked_points or len(clicked_points) < 2:
        return "Click two points on the graph to measure delta."
    dx = abs(clicked_points[1]['x'] - clicked_points[0]['x'])
    dy = abs(clicked_points[1]['y'] - clicked_points[0]['y'])
    return f"Delta: ΔX = {dx:.2f} min, ΔY = {dy:.2f}"


if __name__ == "__main__":
    app.run(debug=True)
