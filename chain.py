"""Acquire and handle option chain data from a source (just Yahoo! Finance for now)."""
import datetime as dt
import pandas as pd
import yahooquery as yq

from option import OptionContract

pd.set_option('display.max_rows', None)
pd.options.display.width = None
pd.options.mode.chained_assignment = None  # default='warn'


def _get_chain_yahoo(symbol: str) -> pd.DataFrame:
    """Gets the full option chain of a ticker symbol from Yahoo! Finance.

    Arguments
    ---------
    symbol : str
        Ticker symbol for which to get the option chain.

    Returns
    -------
    pd.DataFrame
        Flattened option chain. One row for every single option available for the ticker symbol.

    Note
    ----
    Similar functions can be implemented like this for different data sources, but will require significant refactoring
    of get_chain() and split_chain().
    """
    #
    # Get data
    #
    symbol = symbol.upper()
    ticker = yq.Ticker(symbol)
    S = ticker.price[symbol]['regularMarketPrice']
    raw_chain = ticker.option_chain.loc[symbol]
    #
    # Pull out the data we want and rename it
    # The rest of the available data from Yahoo is:
    #       contractSymbol, currency, change, percentChange, contractSize, inTheMoney
    #
    chain = pd.DataFrame()
    chain['expiration'] = [str(x[0]).split(' ')[0] for x in raw_chain.index]
    chain['opt_type'] = [x[1].replace('s', '') for x in raw_chain.index]
    columns = {
        'strike': 'strike', 'lastPrice': 'last', 'bid': 'bid', 'ask': 'ask', 'inTheMoney': 'itm',
        'volume': 'volume', 'openInterest': 'open interest', 'lastTradeDate': 'last traded',
        'impliedVolatility': 'YahooIV'
    }
    for yahoo_name, our_name in columns.items():
        chain[our_name] = raw_chain[yahoo_name].values
    chain.insert(0, 'S', S)
    chain.insert(0, 'symbol', symbol)
    return chain


def get_chain(
    symbol: str, current_datetime, chain: pd.DataFrame = None, src: str = 'yahoo', r: float = 0, q: float = 0,
    print_times: bool = False
) -> pd.DataFrame:
    """Doc"""
    if chain is None:
        #
        # Check data source and get data
        #
        symbol = symbol.upper()
        data_functions = {
            'yahoo': _get_chain_yahoo
        }
        if src not in data_functions:
            raise ValueError(f'source must be one of: {", ".join(list(data_functions.keys()))} (not {src})')
        try:
            now = dt.datetime.now()
            chain = data_functions[src](symbol)
            if print_times:
                print(f'data for {symbol} acquired from {src} in {dt.datetime.now() - now}')
        except Exception as e:
            print(f'Could not get data for {symbol} from {src}: {e}')
    #
    # Calculate IV and the greeks according to BSM IV of various option prices
    #
    now = dt.datetime.now()
    chain.insert(7, 'mid', (chain['bid'] + chain['ask']) / 2)
    for Vtype in ['YahooIV', 'bid', 'ask', 'mid', 'last']:
        chain[f'obj_{Vtype}'] = chain.apply(
            lambda df: OptionContract(
                df['opt_type'],
                df['expiration'],
                df[Vtype] if Vtype != 'YahooIV' else 0,
                df['S'],
                df['strike'],
                r, q,
                current_datetime=current_datetime,
                IV=None if Vtype != 'YahooIV' else df['YahooIV'],
                price_type=Vtype
            ),
            axis=1
        )
        for param in [
            'itm', 'IV', 'delta', 'vega', 'theta', 'rho', 'epsilon', 'Lambda', 'gamma', 'vanna', 'charm', 'vomma',
            'veta', 'phi', 'speed', 'zomma', 'color', 'ultima', 'dualdelta', 'dualgamma'
        ]:
            if [Vtype, param] not in [['YahooIV', 'itm'], ['YahooIV', 'IV']]:
                chain[f'{param}_{Vtype}'] = chain[f'obj_{Vtype}'].apply(lambda x: eval(f'x.{param}'))

        del chain[f'obj_{Vtype}']
    if print_times:
        print(
            f'bid, ask, mid, last, and {src} IV/greeks (1st/2nd/3rd order) calculated for {len(chain)} ' +
            f'options in {dt.datetime.now() - now}'
        )
    return chain


def split_chain(chain: pd.DataFrame, print_times: bool = False) -> list:
    """Split an option chain by price type, option type, and moneyness.

    Arguments
    ---------
    chain       : pd.DataFrame
        Full option chain with all calculations of all price types.
    print_times : bool
        Print how long it took to split up the data.

    Returns
    -------
    [pd.DataFrame]*16
        [
            chain_mid_calls_itm,     chain_mid_calls_otm,     chain_mid_puts_itm,         chain_mid_puts_otm,
            chain_bid_calls_itm,     chain_bid_calls_otm,     chain_bid_itm_puts_itm,     chain_bid_puts_otm,
            chain_ask_calls_itm,     chain_ask_calls_otm,     chain_ask_itm_puts_itm,     chain_ask_puts_otm,
            chain_last_calls_itm,    chain_last_calls_otm,    chain_last_itm_puts_itm,    chain_last_puts_otm
            chain_YahooIV_calls_itm, chain_YahooIV_calls_otm, chain_YahooIV_itm_puts_itm, chain_YahooIV_puts_otm
        ]

    Note
    ----
    Output DataFrames only have parameters that will be plotted.

    PARAMS = [
    'price', 'volume', 'open interest', 'last traded', 'IV', 'delta', 'vega', 'theta',
    'rho', 'epsilon', 'Lambda', 'gamma', 'vanna', 'charm', 'vomma', 'veta', 'phi', 'speed', 'zomma', 'color', 'ultima',
    'dualdelta', 'dualgamma'
    ]
    """
    def drop_rename_cols(price_type: str, opt_type: str, itm: bool, chain: pd.DataFrame) -> pd.DataFrame:
        """Helper function to split up the data.

        Arguments
        ---------
        price_type : str
            can be 'mid', 'bid', 'ask', or 'last'
        opt_type   : str
            can be 'call' or 'put'
        itm        : bool
            True = itm, False = otm
        chain      : pd.DataFrame
            full option chain

        Returns
        -------
        [pd.DataFrame]*16
        """
        if price_type != 'YahooIV':
            partial_chain = chain[(chain['opt_type'] == opt_type) & (chain[f'itm_{price_type}'] == itm)]

            for col in partial_chain.columns:
                #
                # Delete all columns that aren't basic info or related to the price type
                #
                if (col not in ['strike', 'expiration', 'volume', 'open interest', 'last traded', 'opt_type']) and\
                   (price_type not in col):
                    del partial_chain[col]
                else:
                    #
                    # Rename columns according to the parameters in the Dash app
                    #
                    if col == price_type:
                        partial_chain.rename(columns={price_type: 'price'}, inplace=True)
                    else:
                        partial_chain.rename(columns={col: col.replace(f'_{price_type}', '')}, inplace=True)
            #
            # Delete itm because we don't need it anymore
            #
            del partial_chain['itm']
            partial_chain.reset_index(drop=True, inplace=True)
        else:
            partial_chain = chain[(chain['opt_type'] == opt_type) & (chain['itm'] == itm)]

            for col in partial_chain.columns:
                #
                # Delete all columns that aren't basic info or related to the price type
                #
                #
                # Rename columns according to the parameters in the Dash app
                #
                if col == 'mid':
                    partial_chain.rename(columns={'mid': 'price'}, inplace=True)
                elif col == 'YahooIV':
                    partial_chain.rename(columns={col: col.replace('Yahoo', '')}, inplace=True)
                elif (col not in ['strike', 'expiration', 'volume', 'open interest', 'last traded', 'opt_type', 'mid']) \
                        and (price_type not in col):
                    del partial_chain[col]
                else:
                    partial_chain.rename(columns={col: col.replace(f'_{price_type}', '')}, inplace=True)
            #
            # Delete itm because we don't need it anymore
            #
            # del partial_chain['itm']
            partial_chain.reset_index(drop=True, inplace=True)

        return partial_chain

    now = dt.datetime.now()
    datatypes = [
        ['mid',  'call', True],    ['mid',  'call', False],    ['mid',  'put', True],    ['mid',  'put', False],
        ['bid',  'call', True],    ['bid',  'call', False],    ['bid',  'put', True],    ['bid',  'put', False],
        ['ask',  'call', True],    ['ask',  'call', False],    ['ask',  'put', True],    ['ask',  'put', False],
        ['last', 'call', True],    ['last', 'call', False],    ['last', 'put', True],    ['last', 'put', False],
        ['YahooIV', 'call', True], ['YahooIV', 'call', False], ['YahooIV', 'put', True], ['YahooIV', 'put', False]
    ]
    dataframes = [
        drop_rename_cols(data[0], data[1], data[2], chain) for data in datatypes
    ]
    dataframes = [df[df['IV'] > 0] for df in dataframes]
    dataframes = [df[df['price'] > 0] for df in dataframes]
    if print_times:
        print(f'time to split up data: {dt.datetime.now() - now}')

    return dataframes


if __name__ == '__main__':
    base_chain = _get_chain_yahoo('AAPL')
    calc_chain = get_chain(
        'PLTR', current_datetime=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S'),
        r=0.1
    )
    chains = split_chain(calc_chain)
    test = 1
