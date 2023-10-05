import sys
import random
import time
import yaml
import os
import argparse
import dateutil.parser
import pytz
from tzlocal import get_localzone
import tqdm
from datetime import datetime
import pandas as pd
import pathlib
import requests
import json
import dash
from dash import dcc, dash_table, html, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash.dash_table.Format as ddtf
import dash.dash_table.FormatTemplate as ddtft
import hashlib
from textwrap import dedent

# import robot
#import exchange

import logging
import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


# Custom remote librairies
import mosaic as moc
# import mosaic.utils as mut
# import mosaic.bot as mbo
# import mosaic.database as mdd
# import mosaic.decision_model as mdm
# import mosaic.invest_model as mim
import mosaic.db as mdb
import mosaic.trading as mtr
import mosaic.utils as mtu
    

# App config
# ----------
app_config = dict(
    app_name_short="MOSAIC Monitor",
    author="Developed by EdgeMind (www.edgemind.net) 2023-",
    version="1.0.0",
    update_rate=1000*60*30, # in ms
    log_dir=os.path.dirname(__file__),
    line_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
)


# CLI parameters management
# -------------------------
APP_ARG_PARSER = argparse.ArgumentParser(
    description=app_config["app_name_short"] + " " + app_config["version"])

# APP_ARG_PARSER.add_argument(
#     type=str,
#     dest='study_config_filename',
#     help='System parameters (XLS format).')

    
APP_ARG_PARSER.add_argument(
    '-f', '--app-config-filename',
    dest='app_config_filename',
    action='store',
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help='Application configuration filename.')

APP_ARG_PARSER.add_argument(
    '-p', '--progress',
    dest='progress_mode',
    action='store_true',
    default=app_config.get("progress_mode", False),
    help='Show progress bar in the console.')

APP_ARG_PARSER.add_argument(
    '-v', '--verbose',
    dest='verbose_mode',
    action='store_true',
    default=app_config.get("verbose_mode", False),
    help='Display log information on stardard output.')

APP_ARG_PARSER.add_argument(
    '-d', '--debug',
    dest='debug_mode',
    action='store_true',
    default=app_config.get("debug_mode", False),
    help='Display debug on stardard output.')


APP_INPUT_ARGS = APP_ARG_PARSER.parse_args()
app_config.update(vars(APP_ARG_PARSER.parse_args()))

app_config_path = pathlib.Path(app_config["app_config_filename"])

if not app_config_path.is_file():
    app_config_path.touch()

with open(str(app_config_path), 'r', encoding="utf-8") as yaml_file:
    try:
        app_config_ext = yaml.load(yaml_file,
                                   Loader=yaml.SafeLoader)
        if app_config_ext is None:
            app_config_ext = {}
        
    except yaml.YAMLError as exc:
        logging.error(exc)

app_config.update(**app_config_ext)

# Logging configuration
logger = logging.getLogger(__name__)
if app_config.get("verbose_mode"):
    logger.setLevel(logging.INFO)
if app_config.get("debug_mode"):
    logger.setLevel(logging.DEBUG)

log_filename = os.path.join(app_config.get("log_dir", "."), "app.log")
file_handler = logging.FileHandler(log_filename)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]\n%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
    
# Application data
APP_DATA = dict(
    db=moc.ObjMOSAIC.from_dict(app_config["db"]),
    money_fmt=ddtf.Format(),
    pct1_fmt=ddtf.Format(precision=1, scheme=ddtf.Scheme.percentage_rounded),
    pct2_fmt=ddtf.Format(precision=2, scheme=ddtf.Scheme.percentage_rounded),
    pct4_fmt=ddtf.Format(precision=4, scheme=ddtf.Scheme.percentage_rounded),
    perf_fmt=ddtf.Format(precision=4, scheme=ddtf.Scheme.fixed),
)

# Try to connect the APP with bots DB
APP_DATA["db"].connect()

# initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    prevent_initial_callbacks='initial_duplicate',
)

# Utility functions
# -----------------
def build_tooltip(data, field_specs={}, field_sep="\n\n"):
    tooltip_strlist = []
    for k, v in field_specs.items():
        value = mtu.dict_to_yaml_string(data[k])
        field_cur = f"{v}: {value}"
        tooltip_strlist.append(field_cur)
        
    return field_sep.join(tooltip_strlist)

def get_bots_from_db():
    bots = {
        bot["uid"]: bot
        for bot in APP_DATA["db"].get(endpoint="bots")
    }

    return bots

def plotly_create_empty_fig(message="No Data"):
    return {
        'data': [],
        'layout': go.Layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text=message,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        size=64
                    )
                )
            ]
        )
    }


def hash_sha1_int(s):
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16)


def asset_perf_text(asset_perf):
    asset_str = \
        f"""
        <b>Symbol: {asset_perf['symbol']}</b><br>
        Started at: {asset_perf['dt_start']}<br>
        Date: {asset_perf['dt']}<br>
        Performance: {asset_perf['performance']:.4f}
        """
    return dedent(asset_str)


def bot_perf_text(pf):
    pf_str = \
        f"""
        <b>Bot: {pf['bot_uid_short'].upper()}</b><br>
        Date: {pf['dt']}<br>
        Performance: {pf['performance']:.4f}<br>
        """
    return dedent(pf_str)


def order_text(order):
    base_name = order['symbol'].split("/")[0]
    quote_name = order['symbol'].split("/")[1]
    
    od_str = \
        f"""
        <b>{order['side'].upper()} order</b><br>
        UID: {order['uid']}<br>
        Date: {order['dt_closed']}<br>
        Quote price: {mtu.fmt_currency(order['quote_price'])} {quote_name}<br>
        Quote amount: {mtu.fmt_currency(order['quote_amount'])} {quote_name}<br>
        Base amount: {mtu.fmt_currency(order['base_amount'])} {base_name}<br>
        Fees: {order['fees']['value']} {order['fees']['asset']}<br>
        """

    return dedent(od_str)


# Basic components
# ----------------
APP_COMP = {}

comp_name = "bots_update_btn"
APP_COMP[comp_name] = \
    dbc.Button('Update',
               id=comp_name,
               n_clicks=0, color='primary')

comp_name = "bots_table"
APP_COMP[comp_name] = dash_table.DataTable(
    id=comp_name,
    row_selectable='multi',
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    page_action="native",
    page_current=0,
    page_size=10,
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    style_cell={
        "maxWidth": "125px",
        "minWidth": "20px",
        "textOverflow": "ellipsis",
    },
    tooltip_delay=0,
    tooltip_duration=None,
    merge_duplicate_headers=True,
)

comp_name = "orders_selected_table"
APP_COMP[comp_name] = dash_table.DataTable(
    id=comp_name,
    # row_selectable='multi',
    filter_action="native",
    sort_action="native",
    sort_mode="multi",
    page_action="native",
    page_current=0,
    page_size=10,
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    style_cell={
        "maxWidth": "100px",
        "minWidth": "20px",
        "textOverflow": "ellipsis",
    },
)

comp_name = "bots_perf_time_graph"
APP_COMP[comp_name] = dcc.Graph(id=comp_name)

comp_name = "bots_trades_time_graph"
APP_COMP[comp_name] = dcc.Graph(id=comp_name)

# Compound components
# -------------------
comp_name = "bots_selected_card"
APP_COMP[comp_name] = dbc.Card(
    [
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H3("Bot sessions"),
                        width={"size": 10, "offset": 0}),
                dbc.Col(APP_COMP["bots_update_btn"],
                        width={"size": 1, "offset": 0}),
            ])
        ]),
        dbc.CardBody([
            APP_COMP["bots_table"]
        ])
    ])

comp_name = "orders_selected_card"
APP_COMP[comp_name] = dbc.Card(
    [
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H3("Orders"),
                        width={"size": 10, "offset": 0}),
            ])
        ]),
        dbc.CardBody([
            APP_COMP["orders_selected_table"]
        ])
    ])


comp_name = "bots_perf_card"
APP_COMP[comp_name] = dbc.Card(
    [
        dbc.CardHeader([
            html.H3("Bots performance")
        ]),
        #     dbc.Row([
        #         dbc.Col(
        #             html.H3("Bots performance"),
        #             width={"size": 10, "offset": 0}),
        #     ])
        # ]),
        dbc.CardBody([
            dbc.Row(
                [
                    dbc.Col(
                        APP_COMP["bots_perf_time_graph"],
                        width={"size": 8, "offset": 0}
                    ),
                    dbc.Col(
                        APP_COMP["bots_trades_time_graph"],
                        width={"size": 4, "offset": 0}
                    ),
                ]
            )
        ]),
    ])


# comp_name = "bots_indics_card"
# APP_COMP[comp_name] = dbc.Card(
#     [
#         dbc.CardHeader([
#             dbc.Row([
#                 dbc.Col(html.H3("Trading indicators"),
#                         width={"size": 10, "offset": 0}),
#             ])
#         ]),
#         dbc.CardBody([
#             APP_COMP["bots_trades_time_graph"]
#         ])
#     ])


# General Layout
# --------------
bots_dash = html.Div([
    dbc.Row([
        
        dbc.Col([
            APP_COMP["bots_perf_card"],
            #APP_COMP["bots_indics_card"],
        ], width={"size": 12, "offset": 0}),
        
        dbc.Col([
             APP_COMP["bots_selected_card"],
        ], width={"size": 12, "offset": 0}),
        
    ], justify="left")
], id="bots_dash")


# bot_monitoring_tab = dbc.Tab(
#     html.Div(
#         [
#             html.Br(),
#             dbc.Row([
#                 html.H2('Selected bots'),
#                 dbc.Col(
#                     dash_table.DataTable(
#                         id='bots_selected_table',
#                         # row_selectable='multi',
#                         filter_action="native",
#                         sort_action="native",
#                         sort_mode="multi",
#                         page_action="native",
#                         page_current=0,
#                         page_size=10,
#                     ),
#                     width={"size": 3, "offset": 0},
#                 ),
#             ], justify="center"),
#             dbc.Row([
#                 html.H2('Orders'),
#                 dbc.Col(
#                     dash_table.DataTable(
#                         id='orders_selected_table',
#                         # row_selectable='multi',
#                         filter_action="native",
#                         sort_action="native",
#                         sort_mode="multi",
#                         page_action="native",
#                         page_current=0,
#                         page_size=10,
#                         style_cell={
#                             "maxWidth": "100px",
#                             "minWidth": "20px",
#                             "textOverflow": "ellipsis",
#                         },
#                     ),
#                     width={"size": 3, "offset": 0},
#                 ),
#                 dbc.Col(
#                     dcc.Graph(id='bots_perf_time_graph'),
#                     width={"size": 7, "offset": 1},
#                 ),
#             ], justify="center"),
#         ]),
#     id="bot_monitoring_tab",
#     label="Monitoring",
# )

APP_STORES = {}
store_name = "bots"
APP_STORES[store_name] = dcc.Store(
    id=store_name,
    data={},
)
store_name = "bots_selected"
APP_STORES[store_name] = dcc.Store(
    id=store_name,
    data={},
)
store_name = "bots_perf_selected"
APP_STORES[store_name] = dcc.Store(
    id=store_name,
    data={},
)
store_name = "orders_selected"
APP_STORES[store_name] = dcc.Store(
    id=store_name,
    data={},
)

app.layout = html.Div(
    list(APP_STORES.values()) +
    [
        dcc.Interval(
            id='bots_update',
            interval=app_config["update_rate"],  # in milliseconds
            n_intervals=0,
        ),

        
        html.H1('MOSAIC Monitor',
                style={'textAlign': 'center'}),
        html.Hr(),
        bots_dash,
#         dbc.Tabs(
#             [
#                 bots_tab,
# #                bot_monitoring_tab,
#             ]
#         ),
    ]
)


# Callbacks
# ---------

@app.callback(
    Output('bots', 'data', allow_duplicate=True),
    [Input('bots_update_btn', 'n_clicks')],
    prevent_initial_call=True, 
)
def bots_update_on_click(n_clicks):
    bots = {}
    if n_clicks > 0:    
        bots = {
            bot["uid"]: bot
            for bot in APP_DATA["db"].get(endpoint="bots")
        }

    return bots


@app.callback(
    Output('bots', 'data'),
    [Input('bots_update', 'n_intervals')],
    prevent_initial_call=True,
)
def botsession_table_update(n_intervals):
    logger.debug(f"botsession_table_update: {n_intervals}")
    return {
        bot["uid"]: bot
        for bot in APP_DATA["db"].get(endpoint="bots")
    }

@app.callback(
    [
        Output('bots_table', 'columns'),
        Output('bots_table', 'data'),
        Output('bots_table', 'tooltip_data'),
    ],
    # Output('bots_table', 'children'),
    [Input('bots', 'data')],
)
def bots_table_update(bots):

    bots_data = []
    bots_tooltip = []
    bots_columns = [
        {"id": "uid", "name": ("", "UID")},
        {"id": "name", "name": ("", "Name")},
        {"id": "mode", "name": ("", "Mode")},
        {"id": "ds_trading__symbol", "name": ("Trading context", "Symbol")},
        {"id": "ds_trading__timeframe", "name": ("Trading context", "tf")},
        {"id": "ds_trading__dt_start", "name": ("Trading context", "Start")},
        {"id": "ds_trading__dt_end", "name": ("Trading context", "End")},
        {"id": "portfolio__performance", "name": ("", "Perf."),
         "type": "numeric", "format": APP_DATA["perf_fmt"]},
        {"id": "progress", "name": ("", "Progress"),
         "type": "numeric", "format": APP_DATA["pct1_fmt"]},
        {"id": "status", "name": ("", "Status")},
        ]
    bots_tooltip_specs = \
        [
            {"id": "uid", "names": {"decision_model": "DM"}},
        ]
    
    
    if bots:
        var = [specs["id"] for specs in bots_columns]

        for data_ori in bots.values():
            data = mtu.flatten_dict(data_ori, join_key="__", level=1)
            bot_specs = {key: mtu.parse_value(data[key]) for key in var}
            bot_specs["id"] = bot_specs["uid"]
            bots_data.append(bot_specs)

            bots_tooltip_cur = \
                {
                    t_specs["id"]: {"value": build_tooltip(data_ori,
                                                           field_specs=t_specs["names"]),
                                    "type": "markdown"}
                    for t_specs in bots_tooltip_specs
                }
            bots_tooltip.append(bots_tooltip_cur)
                                        
                
    return bots_columns, bots_data, bots_tooltip


@app.callback(
    Output('bots_selected', 'data', allow_duplicate=True),
    Output('bots_perf_selected', 'data', allow_duplicate=True),
    Output('orders_selected', 'data', allow_duplicate=True),
    Input('bots_table', 'selected_row_ids'),
    Input('bots', 'data'),
)
def bots_selected_update(bot_uid_selected, bots):
    bots_selected = {}
    bots_perf_selected = {}
    orders_selected = {}
    bot_uid_selected = bot_uid_selected if bot_uid_selected else []
    for uid in bot_uid_selected:
        bots_selected[uid] = bots[uid]
        bots_perf_selected[uid] = \
            APP_DATA["db"].get(endpoint="portfolio",
                               filter={"bot_uid": uid})
        orders_selected[uid] = \
            APP_DATA["db"].get(endpoint="orders",
                               filter={"bot_uid": uid})

    return bots_selected, bots_perf_selected, orders_selected


# @app.callback(
#     [Output('orders_selected_table', 'columns'),
#      Output('orders_selected_table', 'data')],
#     # Output('bots_table', 'children'),
#     [Input('orders_selected', 'data')],
#     prevent_initial_call=True,
# )
# def orders_selected_table_update(orders_selected):

#     orders_selected_data = []
#     orders_selected_columns = [
#         {"id": "bot_uid", "name": "Bot"},
#         {"id": "uid", "name": "UID"},
#         {"id": "cls", "name": "Order type"},
#         {"id": "side", "name": "Side"},
#         {"id": "status", "name": "Status"},
#         {"id": "dt_closed", "name": "Closed DT"},
#         {"id": "base_amount", "name": "Base amount",
#          "type": "numeric", 'format': APP_DATA["money_fmt"]},
#         {"id": "quote_amount", "name": "Quote amount",
#          "type": "numeric", 'format': APP_DATA["money_fmt"]},
#         {"id": "quote_price", "name": "Quote price",
#          "type": "numeric", 'format': APP_DATA["money_fmt"]},
#         {"id": "fees__value", "name": "Fees",
#          "type": "numeric", 'format': APP_DATA["money_fmt"]},
#         ]
        
#     if orders_selected:

#         var = [specs["id"] for specs in orders_selected_columns]

#         for orders_list in orders_selected.values():
#             for order in orders_list:
#                 order_flat = mtu.flatten_dict(order, join_key="__", level=1)
#                 order_specs = {key: mtu.parse_value(order_flat[key]) for key in var}
#                 order_specs["id"] = order_specs["uid"]
#                 orders_selected_data.append(order_specs)

#     return orders_selected_columns, orders_selected_data


def create_bots_perf_time_graph(bot_perf_selected_df,
                                orders_selected_df,
                                ):
    fig = go.Figure()

    # Fix the y-axis to 0 and 5% more than the max value
    y_min = 0
    y_max = bot_perf_selected_df['performance'].max()*1.2
    # Get the min and max dates
    x_min = bot_perf_selected_df['dt_start'].min()
    x_max = bot_perf_selected_df['dt_end'].max()

    # Add traces for each bot_uid_short
    for bot_name, bot_perf_df in bot_perf_selected_df.groupby('bot_uid_short'):

        bot_uid = bot_perf_df["bot_uid"].iloc[0]

        line_color = \
            app_config["line_colors"][hash_sha1_int(bot_uid) %
                                      len(app_config["line_colors"])]

        pf_text = \
            [bot_perf_text(pf) for pf in bot_perf_df.to_dict("records")]
        fig.add_scatter(x=bot_perf_df['dt'],
                        y=bot_perf_df['performance'],
                        mode='markers+lines',
                        name="performance",
                        legendgrouptitle_text=bot_name,
                        legendgroup=bot_name,
                        hovertext=pf_text,
                        hoverinfo="text",
                        line=dict(width=5,
                                  color=line_color,
                                  ))

        # Add buy markers
        if len(orders_selected_df) > 0:
            orders_df = orders_selected_df.loc[[bot_uid]]

            buy_orders_df = orders_df[orders_df['side'] == 'buy']
            buy_orders_text = \
                [order_text(od) for od in buy_orders_df.to_dict("records")]
            fig.add_trace(go.Scatter(
                x=buy_orders_df['dt_closed'],
                y=buy_orders_df['performance'],
                mode='markers',
                name='Buy orders',
                hovertext=buy_orders_text,
                hoverinfo="text",
                marker=dict(size=13,
                            symbol='triangle-right',
                            color=line_color,
                            opacity=0.7,
                            ),
                legendgroup=bot_name,
            ))

            # Add buy markers
            sell_orders_df = orders_df[orders_df['side'] == 'sell']
            sell_orders_text = [order_text(od) for od in sell_orders_df.to_dict("records")]
            fig.add_trace(go.Scatter(
                x=sell_orders_df['dt_closed'],
                y=sell_orders_df['performance'],
                mode='markers',
                name='Sell orders',
                hovertext=sell_orders_text,
                hoverinfo="text",
                marker=dict(size=13,
                            symbol='triangle-left',
                            color=line_color,
                            opacity=0.7,
                            ),
                legendgroup=bot_name,
            ))

    quote_price_df = \
        bot_perf_selected_df[["symbol", "dt_start", "dt", "quote_price"]]
    for (symbol, dt_start), qp_df in quote_price_df.groupby(["symbol","dt_start"]):
        qp_df = qp_df.sort_values(by="dt")
        qp_df["performance"] = qp_df["quote_price"]/qp_df["quote_price"].iloc[0]

        asset_name = f"{symbol} ({dt_start.date()})"

        line_color = \
            app_config["line_colors"][hash_sha1_int(asset_name) %
                                      len(app_config["line_colors"])]

        qp_text = \
            [asset_perf_text(qp) for qp in qp_df.to_dict("records")]
        fig.add_scatter(x=qp_df['dt'],
                        y=qp_df['performance'],
                        mode='lines',
                        name=asset_name,
                        legendgroup="Asset",
                        hovertext=qp_text,
                        hoverinfo="text",
                        line=dict(width=2,
                                  dash="dash",
                                  color=line_color,
                                  ))


    # Update line width and axis
    #fig.update_traces(line=dict(width=5))
    fig.update_layout(
        margin=dict(t=30),
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(range=[x_min, x_max]),
        showlegend=True,
    )

    return fig



def create_trades_time_graph(orders_selected_df):

    idx_buy = orders_selected_df["side"] == "buy"
    buy_time_df = orders_selected_df.loc[idx_buy, "dt_closed"]
    time_to_trade_df = buy_time_df.groupby(level=0).diff()\
                                                   .rename("value")\
                                                   .dropna()\
                                                   .reset_index()
    time_to_trade_df["indic"] = "Time between trades"

    # Drop consecutive duplicates
    idx_no_dup = orders_selected_df["side"].shift() != orders_selected_df["side"]
    buy_sell_time_df = orders_selected_df.loc[idx_no_dup, "dt_closed"]
    trade_duration_df = buy_sell_time_df.groupby(level=0).diff()\
                                                         .rename("value")\
                                                         .dropna()\
                                                         .reset_index()
    trade_duration_df["indic"] = "Trade duration"
    
    trades_time_df = pd.concat([
        time_to_trade_df,
        trade_duration_df,
        ], axis=0, ignore_index=True)

    trades_time_df["value"] = trades_time_df["value"].dt.total_seconds()/3600

    color_discrete_map = {
        bot_uid: app_config["line_colors"][hash_sha1_int(bot_uid) %
                                           len(app_config["line_colors"])]
        for bot_uid in trades_time_df["bot_uid"].unique()
    }
    
    fig = \
        px.box(trades_time_df,
               x="indic",
               y="value",
               color="bot_uid",
               color_discrete_map=color_discrete_map,
               labels={
                   "bot_uid": "Bot ID",
                   "value": "Duration (in hours)",
                   "indic": "",
               })

    fig.update_layout(
        margin=dict(t=30),
        showlegend=False,
    )

    return fig


@app.callback(
    [
        Output('bots_perf_time_graph', 'figure'),
        Output('bots_trades_time_graph', 'figure')
     ],
    [
        Input('bots_perf_selected', 'data'),
        Input('orders_selected', 'data'),
    ],
    State('bots_selected', 'data'),
    prevent_initial_call=True,
)
def update_bots_perf(bots_perf_selected, orders_selected, bots_selected):

    if bots_perf_selected:
        # Concaténez les dataframes de bots_perf_selected si c'est une liste de DataFrames
        bots_perf_selected_df = pd.concat([
            pd.DataFrame(bot_perf)
            for bot_perf in bots_perf_selected.values()],
            axis=0)

        #ipdb.set_trace()
        bots_perf_selected_df["dt"] = \
            bots_perf_selected_df["dt"].astype("datetime64[ns]")\
                                        .dt.tz_localize("utc")\
                                           .dt.tz_convert(get_localzone())
        bots_perf_selected_df = \
            bots_perf_selected_df.sort_values("dt")
        
        orders_selected_df = pd.concat([
            pd.DataFrame(orders)
            for orders in orders_selected.values()],
            axis=0)
        if len(orders_selected_df) > 0:
            orders_selected_df["dt_open"] = \
                orders_selected_df["dt_open"].astype("datetime64[ns]")\
                                             .dt.tz_localize("utc")\
                                                .dt.tz_convert(get_localzone())
            orders_selected_df["dt_closed"] = \
                orders_selected_df["dt_closed"].astype("datetime64[ns]")\
                                               .dt.tz_localize("utc")\
                                                  .dt.tz_convert(get_localzone())
            orders_selected_df = \
                orders_selected_df.sort_values("dt_closed")

            orders_selected_df = \
                pd.merge_asof(orders_selected_df,
                              bots_perf_selected_df[["bot_uid",
                                                      "dt",
                                                      "performance"]],
                              by="bot_uid",
                              left_on='dt_closed',
                              right_on='dt').set_index("bot_uid")
            fig_trades_time = \
                create_trades_time_graph(orders_selected_df)
        else:
            fig_trades_time = plotly_create_empty_fig("No trades")
            #print(portfolios_selected_df)
        
        bots_selected_df = \
            pd.DataFrame([{
                "uid": bot["uid"],
                "name": bot["name"],
                "symbol": bot["ds_trading"]["symbol"],
                "dt_start": mtu.parse_value(bot["ds_trading"]["dt_start"]),
                "dt_end": mtu.parse_value(bot["ds_trading"]["dt_end"]),
            } for bot in bots_selected.values()]).set_index("uid")
        bots_selected_df.index.name = "bot_uid"
                
        bots_perf_selected_df = \
            bots_perf_selected_df.set_index("bot_uid")\
                                 .join(bots_selected_df,
                                       how="left").reset_index()
        
        bots_perf_selected_df["bot_uid_short"] = \
            bots_perf_selected_df["name"] + "-" + bots_perf_selected_df["bot_uid"].str[:6]

        fig_bots_perf_time = \
            create_bots_perf_time_graph(bots_perf_selected_df,
                                        orders_selected_df)

        return fig_bots_perf_time, fig_trades_time

    else:
        return plotly_create_empty_fig("Please select a bot"), \
            plotly_create_empty_fig("")


# @app.callback(
#     Output('bots_perf_time_graph', 'figure'),
#     [Input('bots_perf_selected', 'data')],
#     State('bots_selected', 'data'),
#     prevent_initial_call=True,
# )
# def update_portfolio_graph(bots_perf_selected, bots_selected):

#     if bots_perf_selected:
#         # Concaténez les dataframes de bots_perf_selected si c'est une liste de DataFrames
#         portfolios_selected_df = pd.concat([
#             pd.DataFrame(portfolio)
#             for portfolio in bots_perf_selected.values()],
#             axis=0)

#         #print(portfolios_selected_df)
        
#         # Créez le graphique de ligne en utilisant plotly.express
#         bots_selected_df = \
#             pd.DataFrame([{
#                 "uid": bot["uid"],
#                 "name": bot["name"],
#                 "dt_start": bot["ds_trading"]["dt_start"],
#                 "dt_end": bot["ds_trading"]["dt_end"],
#             } for bot in bots_selected.values()]).set_index("uid")
#         bots_selected_df.index.name = "bot_uid"

#         portfolios_selected_df = \
#             portfolios_selected_df.set_index("bot_uid")\
#                                   .join(bots_selected_df,
#                                         how="left").reset_index()
        
#         portfolios_selected_df["bot_uid_short"] = \
#             portfolios_selected_df["name"] + "-" + portfolios_selected_df["bot_uid"].str[:6]
#         fig = px.line(portfolios_selected_df,
#                       x='dt',
#                       y='performance',
#                       color="bot_uid_short",
#                       title='Portfolio over Time',
#                       labels={
#                           "dt": "Time",
#                           "performance": "Performance",
#                           "bot_uid_short": "Bot",
#                           },
#                       )

#         # Fix the y-axis to 0 and 5% more than the max value
#         y_min = 0
#         y_max = portfolios_selected_df['performance'].max()*1.5
        
#         # Get the min and max dates
#         x_min = portfolios_selected_df['dt_start'].min()
#         x_max = portfolios_selected_df['dt_end'].max()
        
#         # Update line width and axis
#         fig.update_traces(line=dict(width=5))
#         fig.update_layout(title=None,
#                           margin=dict(t=0),
#                           yaxis=dict(range=[y_min, y_max]),
#                           xaxis=dict(range=[x_min, x_max])) 
#         return fig
#     else:
#         return plotly_create_empty_fig("Please select a bot")

    
# bots_list = APP_DATA["db"].get(endpoint="trading_session")
# ipdb.set_trace()
# bot = mtr.BotTrading.from_dict(bots_list[0])

# Initialization
# --------------

APP_STORES["bots"].data = get_bots_from_db()



if __name__ == "__main__":
    app.run_server(debug=True)
