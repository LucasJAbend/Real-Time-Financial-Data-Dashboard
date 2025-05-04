import csv
import yfinance as yf
from datetime import datetime, timedelta
import concurrent.futures
import time
import logging

# Set up logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

under_20_million_float_and_between_1_and_20_dollars = []

def fetch_float_and_price(stockSymbol):
    try:
        print(f"Fetching data for {stockSymbol}...")  # Log fetching action
        time.sleep(1)  # Delay to avoid hitting API rate limits
        ticker = yf.Ticker(stockSymbol)
        info = ticker.info
        
        float_shares = info.get('floatShares', None)
        current_price = info.get('currentPrice', None)
        
        return stockSymbol, float_shares, current_price
    except Exception as e:
        logging.error(f"Error fetching data for {stockSymbol}: {e}")
        return stockSymbol, None, None

def calculate_3_month_avg_volume(stockSymbol):
    """
    Calculates the 3-month average trading volume for a given stock symbol.
    
    Args:
    stockSymbol (str): The stock symbol to fetch data for.
    
    Returns:
    float: The 3-month average trading volume, or None if data is not available.
    """
    try:
        ticker = yf.Ticker(stockSymbol)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if len(hist) < 60:
            print(f"Not enough data for {stockSymbol} to calculate 3-month average volume.")
            return None
        
        avg_volume_3m = hist['Volume'].mean()
        return avg_volume_3m
    except Exception as e:
        logging.error(f"Error calculating 3-month average volume for {stockSymbol}: {e}")
        return None

def filter_and_calculate_volume():
    """
    Calculates the 3-month average volume for the filtered stocks.
    """
    final_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_stock = {executor.submit(calculate_3_month_avg_volume, stock[0]): stock for stock in under_20_million_float_and_between_1_and_20_dollars}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            stock_symbol, float_shares, current_price = stock
            try:
                avg_volume_3m = future.result()
                if avg_volume_3m is not None:
                    final_list.append((stock_symbol, float_shares, current_price, avg_volume_3m))
            except Exception as e:
                logging.error(f"Error processing stock {stock_symbol}: {e}")

    return final_list

def sanitize_symbol(symbol):
    """
    Sanitizes a stock symbol by replacing invalid characters.
    
    Args:
    symbol (str): The stock symbol to sanitize.
    
    Returns:
    str: The sanitized stock symbol.
    """
    return symbol.replace('^', '-')

def main():
    """
    Main function to read stock symbols from a CSV file, fetch float shares,
    and filter stocks with float shares under 20 million and price between $1 and $20.
    """
    global under_20_million_float_and_between_1_and_20_dollars

    with open('valid_tickers.csv', mode='r') as file:
        heading = next(file)
        csvFile = csv.reader(file)
        
        linecount = 0
        stockCount = 0
        allStockSymbols = []

        for lines in csvFile:
            linecount += 1
            stock_symbol = sanitize_symbol(lines[0])
            allStockSymbols.append(stock_symbol)
        
        print("Starting")
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_symbol = {executor.submit(fetch_float_and_price, symbol): symbol for symbol in allStockSymbols}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    stock_symbol, float_shares, current_price = future.result()
                    if float_shares is not None and current_price is not None:
                        stockCount += 1
                        print(f"{stockCount}/{linecount}", end="\r")

                        if float_shares < 20_000_000 and 2 <= current_price <= 20:
                            under_20_million_float_and_between_1_and_20_dollars.append(
                                (stock_symbol, float_shares, current_price)
                            )
                except Exception as e:
                    logging.error(f"Error processing symbol {symbol}: {e}")
        
        print("\nFiltering and calculating average volumes...")
        under_20_million_float_and_between_1_and_20_dollars = filter_and_calculate_volume()
        print("Done")

if __name__ == "__main__":
    main()
