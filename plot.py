"""doc"""
from chain import split_chain
import pandas as pd
import plotly.graph_objects as go


def plot_surface3d(symbol: str, S: float, chain: pd.DataFrame, data_to_plot: dict):
    """Plot 3d surfaces of itm/otm calls/puts.

    data_to_plot = {
        price_type: 'mid' or 'bid' or 'ask' or 'last'
        opt_type: ['calls', 'puts']
        moneyness: ['itm', 'otm'],
        param: 'price' or 'delta' or 'open interest' or ...
    }

    """
    price_type = data_to_plot['price_type']
    opt_type = data_to_plot['opt_type']
    moneyness = data_to_plot['moneyness']
    param = data_to_plot['param']
    #
    # Get data
    #
    calls_mid_itm,     calls_mid_otm,     puts_mid_itm,     puts_mid_otm, \
    calls_bid_itm,     calls_bid_otm,     puts_bid_itm,     puts_bid_otm, \
    calls_ask_itm,     calls_ask_otm,     puts_ask_itm,     puts_ask_otm, \
    calls_last_itm,    calls_last_otm,    puts_last_itm,    puts_last_otm, \
    calls_YahooIV_itm, calls_YahooIV_otm, puts_YahooIV_itm, puts_YahooIV_otm = split_chain(chain, print_times=True)
    del chain
    x = 'strike'
    y = 'expiration'
    z = param
    fig = go.Figure()
    fig.update_layout(
        title='{} current price: ${:.2f}'.format(symbol.upper(), S),
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
            xaxis_autorange='reversed',
            yaxis_autorange='reversed',
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False
        )
    )

    call_df = pd.DataFrame()
    puts_df = pd.DataFrame()
    for o in opt_type:
        for m in moneyness:
            df = eval(f'{o}_{price_type}_{m}')
            if o == 'calls':
                call_df = pd.concat([call_df, df])
            elif o == 'puts':
                puts_df = pd.concat([puts_df, df])

    if len(call_df) != 0:
        fig.add_trace(
            go.Mesh3d(
                name='calls', x=call_df[x], y=call_df[y], z=call_df[z], intensity=call_df[z],
                colorscale='algae', colorbar=dict(title='call '+param, xanchor='left'), opacity=0.5,
                contour=dict(color='green', show=True, width=10)
            )
        )

    if len(puts_df) != 0:
        fig.add_trace(
            go.Mesh3d(
                name='puts', x=puts_df[x], y=puts_df[y], z=puts_df[z], intensity=puts_df[z],
                colorscale='solar_r', colorbar=dict(title='put '+param, xanchor='right'), opacity=0.5,
                contour=dict(color='red', show=True, width=10)
            )
        )

    fig.update_traces(hovertemplate=f'{x}:' + '%{x} <br>' + f'{y}: ' + '%{y}<br>' + f'{z}: ' + '%{z}')

    return fig


def plot_lines2d(symbol: str, S: float, chain: pd.DataFrame, data_to_plot: dict):
    """Doc"""
    price_type = data_to_plot['price_type']
    opt_type = data_to_plot['opt_type']
    moneyness = data_to_plot['moneyness']
    param = data_to_plot['param']
    #
    # Get data
    #
    calls_mid_itm,     calls_mid_otm,     puts_mid_itm,     puts_mid_otm, \
    calls_bid_itm,     calls_bid_otm,     puts_bid_itm,     puts_bid_otm, \
    calls_ask_itm,     calls_ask_otm,     puts_ask_itm,     puts_ask_otm, \
    calls_last_itm,    calls_last_otm,    puts_last_itm,    puts_last_otm, \
    calls_YahooIV_itm, calls_YahooIV_otm, puts_YahooIV_itm, puts_YahooIV_otm = split_chain(chain, print_times=True)
    del chain

    fig = go.Figure()
    fig.update_layout(
        title='{} current price: ${:.2f}'.format(symbol.upper(), S),
        xaxis_title='strike',
        yaxis_title=param
    )

    for o in opt_type:
        for m in moneyness:
            df = eval(f'{o}_{price_type}_{m}')
            exps = sorted(list(set(df['expiration'])))
            for exp in exps:
                x = df[df['expiration'] == exp]['strike']
                y = df[df['expiration'] == exp][param]
                color = 'green' if o == 'calls' else 'red'
                fig.add_trace(
                    go.Scatter(
                        name=f'{exp} {o}', x=x, y=y, line=dict(color=color, width=1), marker=dict(size=0, opacity=0)
                    )
                )

    return fig
