"""Run this app with `python app.py` and visit http://127.0.0.1:8050/ in your web browser."""
# TODO: add source type: YahooIV
# TODO: make the params header a link that goes to a page that gives more information about the param.
# TODO: display 2D or 3D graph mouse controls as a div above the graph
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import datetime as dt
import plotly.express as px
from typing import Tuple, Callable
from chain import _get_chain_yahoo, get_chain
from plot import plot_lines2d, plot_surface3d
from dash.exceptions import PreventUpdate

'''
Basic setup
'''
'<a href="https://en.wikipedia.org/wiki/Greeks_(finance)" target="_blank">the Greeks</a>'
YAHOO_QUOTE_URL = 'https://finance.yahoo.com/quote'
PARAMS = [
    'price', 'volume', 'open interest', 'last traded', 'IV', 'delta', 'vega', 'theta',
    'rho', 'epsilon', 'Lambda', 'gamma', 'vanna', 'charm', 'vomma', 'veta', 'phi', 'speed', 'zomma', 'color', 'ultima',
    'dualdelta', 'dualgamma'
]
app = dash.Dash(__name__)
server = app.server
app.title = 'Options3D'

div_head = html.Div([
    dcc.Link(
        'Implied volatility', href='https://en.wikipedia.org/wiki/Implied_volatility', target='_blank',
        style={'color': 'blue'}
    ),
    '/',
    dcc.Link(
        'the Greeks', href='https://en.wikipedia.org/wiki/Greeks_(finance)', target='_blank', style={'color': 'blue'}
    ),
    ' are calculated according to the ',
    dcc.Link(
        'Black-Scholes-Merton model',
        href='https://en.wikipedia.org/wiki/Black–Scholes_model', target='_blank', style={'color': 'blue'}
    ),
    ', using option prices from ',
    dcc.Link('Yahoo! Finance', href='https://finance.yahoo.com',
             target='_blank', style={'color': 'blue'}),
    '. Implied volatility is solved for using a modified ',
    dcc.Link(
        'bisection method',
        href='https://en.wikipedia.org/wiki/Bisection_method', target='_blank', style={'color': 'blue'}
    ),
    '.', html.I(
        'notes:  all links open in a new tab, click headers that show "?" on the cursor for help',
        style={'textAlign': 'right'}
    ),
    html.Hr()
], style={'textAlign': 'center'})


'''
Fields
'''


def field(name: str, header: str, content: object, message: str) -> Tuple[html.Div, Callable]:
    """Generates a component and a help dialog callback function for parameter input fields.

    Arguments
    ---------
    name    : str
        Name of the component. Used to set the header id, content id, dialog id, and the id of the entire component
    header  : str
        Text to be displayed in the component header
    content : object
        dash_core_components object
    message : str
        Text to be displayed when the header is clicked

    Returns
    -------
    (html.div, dialog_callback)

    Note
    ----
    Each block of the component is named as follows:
        - component id = f'component-{name}'
        - header id = f'header-{name}'
        - content id = f'content-{name}'
        - dialog id = f'dialog-{name}'
    """
    for argname, argval in {'name': name, 'header': header, 'content': content, 'message': message}.items():
        if argval is None:
            raise ValueError(f'failed to specify div_component {argname}')

    component = html.Div([
        html.H4(children=header,
                id=f'header-{name}', style={'cursor': 'help'}),
        html.Div(children=content, id=f'content-{name}'),
        dcc.ConfirmDialog(message=message, id=f'dialog-{name}')
    ], id=f'component-{name}',
        style={
            'display': 'inline-block', 'border': '2px inset', 'textAlign': 'center', 'margin': '5px', 'padding': '5px'
    }
    )

    @app.callback(
        Output(f'dialog-{name}', 'displayed'),
        Input(f'header-{name}', 'n_clicks')
    )
    def dialog_callback(n_clicks):
        """Callback that displays message"""
        return n_clicks is not None

    return component, dialog_callback


div_datetime, callback_datetime = field(
    name='datetime',
    header='Current Date/Time',
    content=dcc.Input(placeholder='default: current datetime',
                      id='input-datetime'),
    message='''
Current date and time to use as the starting point for time to expiration.
    - calculated in seconds, then converted to years
Useful if it is a weekend, and you want to see things in terms of Friday's close.
MUST be of the format: YYYY-MM-DD HH:MM:SS
'''
)

div_opt_type, callback_dialog_opt_type = field(
    name='opt_type',
    header=dcc.Link(
        'Option Type',
        href='https://www.investopedia.com/terms/o/option.asp', target='_blank', style={'color': 'blue'}
    ),
    content=dcc.Checklist(options=[
        {'label': 'calls', 'value': 'calls'},
        {'label': 'puts', 'value': 'puts'}
    ], value=['calls', 'puts'], labelStyle={'display': 'inline-block'}, id='input-opt_type'),
    message='''
Display data for calls, puts, or both.
Click the heading for more information on calls and puts.
'''
)

div_moneyness, callback_dialog_moneyness = field(
    name='moneyness',
    header=dcc.Link(
        'Moneyness',
        href='https://www.investopedia.com/terms/m/moneyness.asp', target='_blank', style={'color': 'blue'}
    ),
    content=dcc.Checklist(options=[
        {'label': 'ITM', 'value': 'itm'},
        {'label': 'OTM', 'value': 'otm'}
    ], value=['otm'], labelStyle={'display': 'inline-block'}, id='input-moneyness'),
    message='''
Display data for options that are in or out of the money, or both.
Click the heading for more information about moneyness.
'''
)


div_r, callback_dialog_r = field(
    name='r',
    header=dcc.Link(
        'Risk Free Rate',
        href='https://www.investopedia.com/terms/r/risk-freerate.asp', target='_blank', style={'color': 'blue'}
    ),
    content=dcc.Input(placeholder='default: 0', id='input-r'),
    message='''
Use an annualized decimal rate (ex: enter 0.01 for 1%).
Click the heading for more information on the risk free rate.     
'''
)

div_q, callback_dialog_q = field(
    name='q',
    header=dcc.Link(
        'Dividend Yield',
        href='https://www.investopedia.com/terms/d/dividendyield.asp', target='_blank', style={'color': 'blue'}
    ),
    content=dcc.Input(placeholder='default: 0', id='input-q'),
    message='''
Use an annualized decimal rate (ex: enter 0.01 for 1%).     
Click the heading for more information on the dividend yield.  
'''
)

div_symbol, callback_dialog_symbol = field(
    name='symbol',
    header=dcc.Link('Symbol', href=YAHOO_QUOTE_URL, target='_blank',
                    id='link-symbol', style={'color': 'blue'}),
    content=dcc.Input(placeholder='example: AAPL', id='input-symbol'),
    message='''
Ticker symbol of the underlying asset.
Click the heading to go to the Yahoo! page of the entered symbol.
'''
)


@app.callback(
    Output('link-symbol', 'href'),
    Input('input-symbol', 'value')
)
def update_symbol_link(symbol: str):
    """Update the link to Yahoo"""
    if symbol is not None:
        return f'{YAHOO_QUOTE_URL}/{symbol.upper()}'
    else:
        return YAHOO_QUOTE_URL


div_price_type, callback_dialog_price_type = field(
    name='price_type',
    header=dcc.Link(
        'Price Type',
        href='https://www.investopedia.com/terms/q/quoted-price.asp', target='_blank', style={'color': 'blue'}
    ),
    content=dcc.Dropdown(
        options=[
            {'label': 'mid', 'value': 'mid'},
            {'label': 'bid', 'value': 'bid'},
            {'label': 'ask', 'value': 'ask'},
            {'label': 'last', 'value': 'last'},
            {'label': 'YahooIV', 'value': 'YahooIV'}
        ], searchable=False, style={'width': '125px'}, value='mid', id='input-price_type'
    ),
    message='''
Option prices to use for implied volatility calculation.
mid = the price between the bid/ask.
last = the price the last trade was executed at.
YahooIV = use IV provided by Yahoo for calculations.
Click the heading for more information on the types of quoted prices.
'''
)


div_param, callback_dialog_param = field(
    name='param',
    header='Parameter',
    content=dcc.Dropdown(
        options=[
            {'label': param, 'value': param} for param in PARAMS
        ], value='IV', searchable=False, style={'width': '125px'}, id='input-param'
    ),
    message='''
Parameter to plot.
If YahooIV is selected as the price type,
the 'price' option will show the mid price.
'''
)

div_plot_type, callback_dialog_plot_type = field(
    name='plot_type',
    header='Plot Type',
    content=dcc.RadioItems(options=[
        {'label': '2D', 'value': '2d'},
        {'label': '3D', 'value': '3d'}
    ], value='3d', labelStyle={'display': 'inline-block'}, id='input-plot_type'),
    message='''
Plot data in 2 dimensions (as curves) or 3 dimensions (as surfaces).
'''
)

'''
Buttons
'''

div_button_recalculate, callback_dialog_button_recalculate = field(
    name='recalculate',
    header='Recalculate',
    content=html.Button('recalculate', id='button-recalculate'),
    message='''
Recalculates parameters. Note that this does not reacquire data.
Useful if you want to do something like see how changing the: 
    - current datetime
    - risk free rate
    - dividend yield
changes the parameters.
'''
)

div_button_getdata, callback_dialog_button_getdata = field(
    name='getdata',
    header='Get Data',
    content=html.Button('get data', id='button-getdata'),
    message='''
Acquires data and calculates parameters for the entered symbol.
Can also be used as a 'refresh' button to update prices/calculations.
'''
)

div_button_copy, callback_dialog_button_copy = field(
    name='copydata',
    header='Copy Data',
    content=html.Button('copy', id='button-copydata'),
    message='''
Copies all data in memory to the clipboard (in table format).
The data can then be easily pasted into Excel for further investigation.

NOTE: data rows with 0 IV are not shown in the plots, but are still present in the data.
'''
)

div_status, callback_dialog_status = field(
    name='status',
    header='Status',
    content=html.Div(children='no data', id='status-displayed',
                     style={'color': 'red'}),
    message='''
Displays information about the data loaded in memory.
'''
)

div_controls = html.Div([
    div_datetime, div_r, div_q, div_symbol, div_button_getdata, div_button_recalculate, div_button_copy, div_opt_type,
    div_moneyness, div_price_type, div_param, div_plot_type, div_status
], style={'textAlign': 'center', 'display': 'inline-flex'})


'''
App function
'''

div_base_chain = html.Div([
    dcc.Store(data=None, id='data-base_chain'),
    dcc.ConfirmDialog(id='dialog-invalid_inputs_bc')
])

div_calc_chain = html.Div([
    dcc.Store(data=None, id='data-calc_chain'),
    dcc.ConfirmDialog(id='dialog-invalid_inputs_cc')
])


@app.callback(
    Output('status-displayed', 'children'),
    Output('status-displayed', 'style'),
    Input('data-base_chain', 'data')
)
def update_status(data):
    """doc"""
    if data is not None:
        symbol = pd.read_json(data, orient='split')['symbol'].iloc[0]
        return [f'{symbol} data ready', {'color': 'green'}]
    else:
        return ['no data', {'color': 'red'}]


@app.callback(
    Output('data-base_chain', 'data'),
    Output('data-calc_chain', 'data'),
    Output('dialog-invalid_inputs_bc', 'message'),
    Output('dialog-invalid_inputs_bc', 'displayed'),
    Input('button-getdata', 'n_clicks'),
    Input('button-recalculate', 'n_clicks'),
    State('input-datetime', 'value'),
    State('input-r', 'value'),
    State('input-q', 'value'),
    State('input-symbol', 'value'),
    State('data-base_chain', 'data'),
    State('data-calc_chain', 'data')
)
def buttons_getdata_recalculate_clicked(
    n_clicks_get, n_clicks_recalc, dt_input, r_input, q_input, symbol, base_data, calc_data
):
    """doc"""
    #
    # Set defaults
    #
    message = []
    display_dialog = False
    valid_inputs = True
    current_datetime = dt.datetime.strftime(
        dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
    r = 0.0
    q = 0.0
    #
    # See which button was clicked
    #
    if (n_clicks_get is not None) or (n_clicks_recalc is not None):
        ctx_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    else:
        ctx_id = None
        raise PreventUpdate

    print(ctx_id)

    try:
        if dt_input not in ['', None]:
            current_datetime = dt.datetime.strftime(
                dt.datetime.strptime(
                    dt_input, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'
            )
    except Exception as e:
        print(e)
        message += ['incorrect datetime format']
        valid_inputs = False
        display_dialog = True
    try:
        if r_input not in ['', None]:
            r = float(r_input)
    except Exception as e:
        print(e, flush=True)
        message += ['risk free rate must be >= 0']
        valid_inputs = False
        display_dialog = True
    try:
        if q_input not in ['', None]:
            q = float(q_input)
    except Exception as e:
        print(e, flush=True)
        message += ['dividend yield must be >= 0']
        valid_inputs = False
        display_dialog = True
    if symbol in ['', None]:
        message += ['enter a ticker symbol']
        valid_inputs = False
        display_dialog = True
    if ctx_id == 'button-recalculate' and base_data is None:
        message = ['No data to recalculate']
        valid_inputs = False
        display_dialog = True

    if not valid_inputs:
        return [base_data, calc_data, '\n'.join(message), display_dialog]

    if ctx_id == 'button-getdata':
        try:
            #
            # Get/calculate data
            #
            symbol = symbol.upper()
            before = dt.datetime.now()
            base_data = _get_chain_yahoo(symbol)
            after = dt.datetime.now()
            print(
                f'time to acquire {symbol} data from yahoo: {after - before}', flush=True)
            calc_data = get_chain(
                symbol,
                current_datetime=current_datetime,
                r=r, q=q,
                chain=base_data.copy(),
                print_times=True
            )
            base_data = base_data.to_json(date_format='iso', orient='split')
            calc_data = calc_data.to_json(date_format='iso', orient='split')
        except Exception as e:
            print(e, flush=True)
            message += [f'{symbol} is either not valid or not optionable']
            display_dialog = True
    elif ctx_id == 'button-recalculate':
        base_data_df = pd.read_json(base_data, orient='split')
        calc_data = get_chain(
            base_data_df['symbol'].iloc[0],
            current_datetime=current_datetime,
            r=r, q=q,
            chain=base_data_df.copy(),
            print_times=True
        )
        calc_data = calc_data.to_json(date_format='iso', orient='split')

    return [base_data, calc_data, '\n'.join(message), display_dialog]


div_copy = html.Div(id='div-copy', children=[dcc.ConfirmDialog(
    id='dialog-copied', message='data copied to clipboard')])


@app.callback(
    Output('dialog-copied', 'displayed'),
    Input('button-copydata', 'n_clicks'),
    State('data-calc_chain', 'data')
)
def button_copydata_clicked(n_clicks, data):
    """doc"""
    if n_clicks and data is not None:
        pd.read_json(data, orient='split').to_clipboard()
        return True
    return False


df = pd.DataFrame(index=[1])
df['x'] = [1]
df['y'] = [1]
fig_placeholder = px.line(df, x='x', y='y', height=740)

div_graph = html.Div([
    dcc.Graph(figure=fig_placeholder, id='graph')
])


@app.callback(
    Output('graph', 'figure'),
    Input('data-calc_chain', 'data'),
    Input('input-opt_type', 'value'),
    Input('input-moneyness', 'value'),
    Input('input-price_type', 'value'),
    Input('input-param', 'value'),
    Input('input-plot_type', 'value')
)
def plot(calc_chain, opt_type, moneyness, price_type, param, plot_type):
    """doc"""
    fig = fig_placeholder
    valid_inputs = True
    for inp in [calc_chain, opt_type, moneyness, price_type, param]:
        if inp is None:
            valid_inputs = False
            break
    if valid_inputs:
        data_to_plot = {
            'price_type': price_type,
            'opt_type': opt_type,
            'moneyness': moneyness,
            'param': param
        }
        print(json.dumps(data_to_plot, indent=4), flush=True)
        calc_chain = pd.read_json(calc_chain, orient='split')
        symbol = calc_chain['symbol'].iloc[0]
        S = calc_chain['S'].iloc[0]
        if plot_type == '2d':
            fig = plot_lines2d(symbol, S, calc_chain, data_to_plot)
        elif plot_type == '3d':
            fig = plot_surface3d(symbol, S, calc_chain, data_to_plot)

    return fig


app.layout = html.Div([
    div_head, div_controls, div_graph, div_base_chain, div_calc_chain, div_copy
])

if __name__ == '__main__':
    app.run_server(debug=True)
