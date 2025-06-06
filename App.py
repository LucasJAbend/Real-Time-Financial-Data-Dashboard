from datetime import datetime, time, timezone
import pytz
import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from dash import dcc, html
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from flask import Flask, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading

# Create a Flask server for the Dash app
server = Flask(__name__)

# Function to fetch stock data using yfinance
def fetch_stock_data(stockSymbol):
    try:
        ticker = yf.Ticker(stockSymbol)
        info = ticker.info

        if not info:
            raise ValueError(f"No data found for symbol: {stockSymbol}")

        # Convert current time to Eastern Time (ET)
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern).time()

        # Define market session times in ET
        pre_market_open = time(4, 0)  # Pre-market opens at 4:00 AM ET
        market_open = time(9, 30)     # Market opens at 9:30 AM ET
        market_close = time(16, 0)    # Market closes at 4:00 PM ET
        post_market_close = time(20, 0) # Post-market closes at 8:00 PM ET

        # Determine which session we are in: pre-market, regular, or post-market
        if pre_market_open <= now_et < market_open:
            current_price = info.get('currentPrice', None)
            current_volume = info.get('regularMarketVolume', None)
            market_status = 'Pre-Market'
        elif market_open <= now_et < market_close:
            current_price = info.get('currentPrice', None)
            current_volume = info.get('regularMarketVolume', None)
            market_status = 'Regular Market'
        elif market_close <= now_et < post_market_close:
            current_price = info.get('currentPrice', None)
            current_volume = info.get('regularMarketVolume', None)
            market_status = 'Post-Market'
        else:
            # Outside of trading hours
            current_price = None
            current_volume = None
            market_status = 'Closed'

        previous_close = info.get('regularMarketPreviousClose', None)
        float_shares = info.get('floatShares', None)
        average_volume = info.get('averageVolume', None)

        if current_price is None or previous_close is None:
            raise ValueError(f"Missing price data for symbol: {stockSymbol}")

        percent_change = ((current_price - previous_close) / previous_close) * 100 if current_price and previous_close else None

        return (stockSymbol, current_price, float_shares, current_volume, average_volume, percent_change, market_status)
    except Exception as e:
        print(f"Error fetching data for {stockSymbol}: {e}")
        return (stockSymbol, None, None, None, None, None, 'Error')

# Function to load stock symbols from 'stocks.csv'
def load_stock_symbols_from_file():
    filename = 'filtered_stocks.csv'
    try:
        df = pd.read_csv(filename)
        # Assuming 'stocks.csv' has a column named 'Stock Symbol' or similar
        # Adjust 'Stock Symbol' if your CSV uses a different column name for symbols
        if 'Stock Symbol' in df.columns:
            # It's good practice to ensure these columns exist even if they are empty
            # to prevent KeyError if the CSV is missing them
            for col in ['Float Shares', '3-Month Average Volume']:
                if col not in df.columns:
                    df[col] = None # Or np.nan for numerical columns
            return df[['Stock Symbol', 'Float Shares', '3-Month Average Volume']]
        else:
            print(f"Error: '{filename}' must contain a 'Stock Symbol' column.")
            return pd.DataFrame(columns=['Stock Symbol', 'Float Shares', '3-Month Average Volume'])
    except FileNotFoundError:
        print(f"Error: '{filename}' not found. Please create this file with your stock symbols.")
        return pd.DataFrame(columns=['Stock Symbol', 'Float Shares', '3-Month Average Volume'])
    except Exception as e:
        print(f"Error loading stock symbols from '{filename}': {e}")
        return pd.DataFrame(columns=['Stock Symbol', 'Float Shares', '3-Month Average Volume'])

# Function to format numerical values in the DataFrame
def format_numbers(df):
    # Convert back to numeric before formatting, then apply formatting
    for col in ['Current Price', 'Float Shares', 'Current Volume', '3-Month Average Volume', 'Percentage Change']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Current Price'] = df['Current Price'].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    df['Float Shares'] = df['Float Shares'].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
    df['Current Volume'] = df['Current Volume'].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
    df['3-Month Average Volume'] = df['3-Month Average Volume'].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")
    return df

# Function to fetch and update stock data
def fetch_and_update_data():
    initial_df = load_stock_symbols_from_file()

    if initial_df.empty:
        # Return an empty DataFrame with all expected columns if no initial data
        return pd.DataFrame(columns=['Stock Symbol', 'Current Price', 'Float Shares', 'Current Volume', '3-Month Average Volume', 'Percentage Change', 'Market Status'])

    stock_symbols = initial_df['Stock Symbol'].tolist()

    # Fetch stock data in parallel
    num_threads = min(multiprocessing.cpu_count() * 2, 20)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data, symbol): symbol for symbol in stock_symbols}
        results = [future.result() for future in as_completed(future_to_symbol)]

    updated_df = pd.DataFrame(results, columns=['Stock Symbol', 'Current Price', 'Float Shares', 'Current Volume', '3-Month Average Volume', 'Percentage Change', 'Market Status'])

    # Merge the Float Shares and 3-Month Average Volume from the initial_df
    # to ensure consistency if yfinance doesn't return them for some reason
    # Set index for merging, then reset
    updated_df = updated_df.set_index('Stock Symbol')
    initial_df_indexed = initial_df.set_index('Stock Symbol')

    # Use update to prioritize non-null values from updated_df, but fill missing with initial_df
    updated_df = updated_df.combine_first(initial_df_indexed)

    updated_df = updated_df.reset_index()


    # Reformat columns that might have lost their proper type after combine_first/merge
    # The format_numbers function will handle the to_numeric conversion
    updated_df = format_numbers(updated_df)

    # Convert Percentage Change back to numeric for sorting before sorting
    updated_df['Percentage Change'] = pd.to_numeric(updated_df['Percentage Change'], errors='coerce')
    updated_df = updated_df.sort_values(by='Percentage Change', ascending=False)


    updated_df.to_csv('updated_filtered_stocks.csv', index=False)

    return updated_df

# Function to load filtered stocks (now loads the dynamically updated one)
def load_filtered_stocks():
    try:
        df = pd.read_csv('updated_filtered_stocks.csv')
        return df
    except Exception as e:
        print(f"Error loading filtered stocks data: {e}")
        return pd.DataFrame()

# Function to filter stocks based on criteria (still defined, but not used in the main display flow)
def filter_stocks(df):
    # Ensure numerical types before comparison
    # Convert formatted strings back to numeric for filtering
    df['Current Volume_numeric'] = pd.to_numeric(df['Current Volume'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['3-Month Average Volume_numeric'] = pd.to_numeric(df['3-Month Average Volume'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['Percentage Change_numeric'] = pd.to_numeric(df['Percentage Change'], errors='coerce')
    df['Current Price_numeric'] = pd.to_numeric(df['Current Price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['Float Shares_numeric'] = pd.to_numeric(df['Float Shares'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')


    df['Relative Volume'] = df['Current Volume_numeric'] / df['3-Month Average Volume_numeric']

    filtered_df = df[
        (df['Relative Volume'] >= 5) &
        (df['Percentage Change_numeric'] >= 10) &
        (df['Current Price_numeric'] >= 2) &
        (df['Current Price_numeric'] <= 20) &
        (df['Float Shares_numeric'] < 20_000_000)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Drop temporary numeric columns if not needed for further processing
    filtered_df = filtered_df.drop(columns=['Current Volume_numeric', '3-Month Average Volume_numeric',
                                            'Percentage Change_numeric', 'Current Price_numeric',
                                            'Float Shares_numeric', 'Relative Volume'], errors='ignore')

    return filtered_df

# API Endpoint for filtered stocks
@server.route('/api/filtered_stocks')
def filtered_stocks_api(): # Renamed to avoid conflict with the function name
    try:
        df = fetch_and_update_data()
        if df.empty:
            return jsonify({"error": "No data available"}), 404

        # Before converting to dict, ensure numeric types for sorting
        df_for_json = df.copy()
        for col in ['Current Price', 'Float Shares', 'Current Volume', '3-Month Average Volume', 'Percentage Change']:
             # Remove formatting for JSON export if it was applied
            if df_for_json[col].dtype == 'object':
                df_for_json[col] = pd.to_numeric(df_for_json[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('%', ''), errors='coerce')
        result = df_for_json.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        print(f"Error fetching filtered stocks: {e}")
        return jsonify({"error": "Error fetching data"}), 500

# Initialize Dash app
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0), # Refreshes every 60 seconds
    html.H1('Top Gainers of the Day', style={'textAlign': 'center'}),
    html.Div(id='last-updated', style={'textAlign': 'center', 'fontSize': 'small', 'marginBottom': '10px'}),
    dcc.Graph(id='gainers-table')
])

# Callback to update the table
@app.callback(
    [Output('gainers-table', 'figure'),
     Output('last-updated', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_table(n):
    try:
        df = fetch_and_update_data()
        now_et = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %I:%M:%S %p %Z')
        last_updated_text = f"Last Updated: {now_et}"


        if df.empty:
            fig = go.Figure()
            fig.update_layout(title='No Data Available')
            return fig, last_updated_text
        else:
            # Ensure the order of columns matches the header
            # Note: 'Percentage Change' will be formatted as a string by format_numbers
            # If you want to display % sign, add it in format_numbers or here
            display_df = df[['Stock Symbol', 'Current Price', 'Float Shares', 'Current Volume', '3-Month Average Volume', 'Percentage Change', 'Market Status']].copy()

            # Add percentage sign to 'Percentage Change' if it's not already there and is numeric
            if 'Percentage Change' in display_df.columns and pd.to_numeric(display_df['Percentage Change'], errors='coerce').notna().any():
                 display_df['Percentage Change'] = display_df['Percentage Change'].apply(
                     lambda x: f"{pd.to_numeric(x):.2f}%" if pd.notnull(pd.to_numeric(x, errors='coerce')) else "N/A"
                 )


            fig = go.Figure(data=[go.Table(
                header=dict(values=list(display_df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[display_df[col] for col in display_df.columns],
                           fill_color='lavender',
                           align='left'))
            ])

            fig.update_layout(title='Top Gainers of the Day')

            return fig, last_updated_text
    except Exception as e:
        print(f"Error updating table: {e}")
        fig = go.Figure()
        fig.update_layout(title='Error Updating Data')
        last_updated_text = f"Last Updated (Error): {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %I:%M:%S %p %Z')}"
        return fig, last_updated_text

# Run app
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)