import csv
import yfinance as yf
from datetime import datetime, timedelta
import concurrent.futures
import time
import logging

try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate="chrome")
    print("Using curl_cffi session with Chrome impersonation.")
except ImportError:
    import requests
    session = requests.Session()
    print("Using standard requests session (curl_cffi not found).")

# Set up logging
logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_float_and_price(stockSymbol):
    try:
        print(f"Fetching data for {stockSymbol}...")
        time.sleep(1)  # To respect API rate limits
        ticker = yf.Ticker(stockSymbol, session=session)
        info = ticker.info

        float_shares = info.get('floatShares')
        current_price = info.get('currentPrice')

        return stockSymbol, float_shares, current_price
    except Exception as e:
        logging.error(f"Error fetching data for {stockSymbol}: {e}")
        return stockSymbol, None, None

def calculate_3_month_avg_volume(stockSymbol):
    try:
        ticker = yf.Ticker(stockSymbol, session=session)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        hist = ticker.history(start=start_date, end=end_date)

        if len(hist) < 60:
            print(f"Not enough data for {stockSymbol} to calculate 3-month average volume.")
            return None

        return hist['Volume'].mean()
    except Exception as e:
        logging.error(f"Error calculating 3-month average volume for {stockSymbol}: {e}")
        return None

def filter_and_calculate_volume(filtered_stocks):
    final_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(calculate_3_month_avg_volume, symbol): (symbol, float_shares, price)
            for symbol, float_shares, price in filtered_stocks
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            symbol, float_shares, price = futures[future]
            try:
                avg_volume = future.result()
                if avg_volume is not None:
                    final_list.append((symbol, float_shares, price, avg_volume))
                    print(f"Volume calculated for {symbol} ({i}/{len(filtered_stocks)})")
            except Exception as e:
                logging.error(f"Error processing stock {symbol}: {e}")

    return final_list

def sanitize_symbol(symbol):
    return symbol.replace('^', '-')

def main():
    filtered_stocks = []
    all_symbols = []

    with open('valid_tickers.csv', mode='r') as file:
        next(file)  # Skip header
        reader = csv.reader(file)
        for line in reader:
            if line:
                all_symbols.append(sanitize_symbol(line[0]))

    print("Fetching float and price data...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_float_and_price, symbol): symbol for symbol in all_symbols}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            symbol = futures[future]
            try:
                sym, float_shares, price = future.result()
                if float_shares is not None and price is not None:
                    print(f"{i}/{len(all_symbols)}", end="\r")
                    if float_shares < 20_000_000 and 2 <= price <= 20:
                        filtered_stocks.append((sym, float_shares, price))
            except Exception as e:
                logging.error(f"Error processing symbol {symbol}: {e}")

    print("\nFiltering and calculating average volumes...")
    final_results = filter_and_calculate_volume(filtered_stocks)

    print("Writing results to filtered_stocks.csv...")
    with open('filtered_stocks.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['Symbol', 'Float Shares', 'Current Price', '3M Avg Volume'])
        for row in final_results:
            writer.writerow(row)

    print("Done.")

if __name__ == "__main__":
    main()