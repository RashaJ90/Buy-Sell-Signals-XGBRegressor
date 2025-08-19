import argparse
import os
import pandas as pd
from typing import Optional
import sys
from datetime import datetime
import logging

# Add the project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.conf import settings

logger = logging.getLogger(__name__)

def _fetch_yfinance(tickers: list[str], start_date: str, end_date: str, interval: str) -> Optional[dict[str, pd.DataFrame]]:
    """Fetches data from Yahoo Finance, returning a dictionary of DataFrames (one per ticker)."""
    import yfinance as yf
    data_frames = {}

    for ticker in tickers:
        try:
            # Download OHLCV price data
            price_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

            if price_data.empty:
                logger.warning(f"No price data for {ticker}")
                continue

            # Remove ticker name from column names if present (flatten MultiIndex columns)
            if isinstance(price_data.columns, pd.MultiIndex):
                price_data.columns = price_data.columns.get_level_values(0)
            elif price_data.columns.name == ticker:
                # If columns have ticker name, remove it
                price_data.columns.name = None

            # Fetch dividend data
            try:
                stock = yf.Ticker(ticker)
                dividends = stock.dividends

                if not dividends.empty:
                    # Filter dividends within the date range
                    dividends = dividends.loc[start_date:end_date]

                    # Remove timezone information from dividend index to match price_data
                    if dividends.index.tz is not None:
                        dividends.index = dividends.index.tz_localize(None)

                    # Convert dividend index to date only (remove time component)
                    dividends.index = dividends.index.date
                    dividends.index = pd.to_datetime(dividends.index)

                    # Reindex dividends to match price_data index and fill missing with 0
                    aligned_dividends = dividends.reindex(price_data.index, fill_value=0.0)
                    aligned_dividends.name = 'Dividends'

                    # Add dividends column to price data
                    price_data = pd.concat([price_data, aligned_dividends], axis=1)
                else:
                    # No dividends found, add column with zeros
                    price_data['Dividends'] = 0.0

            except Exception as div_error:
                logger.warning(f"Could not fetch dividends for {ticker}: {div_error}")
                # Add dividends column with zeros if dividend fetch fails
                price_data['Dividends'] = 0.0

            # Ensure Dividends column exists and is properly filled
            if 'Dividends' not in price_data.columns:
                price_data['Dividends'] = 0.0
            else:
                # Fill any remaining NaN values with 0
                price_data['Dividends'] = price_data['Dividends'].fillna(0.0)

            price_data.columns = price_data.columns.str.lower()
            data_frames[ticker] = price_data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker} from yfinance: {e}")
            continue

    return data_frames if data_frames else None

def _fetch_quandl(tickers: list[str], start_date: str, end_date: str, interval: str) -> Optional[dict[str, pd.DataFrame]]:
    """Fetches data from Quandl, returning a dictionary of DataFrames."""
    import quandl
    api_key = os.getenv("QUANDL_API_KEY")
    if not api_key:
        logger.error("QUANDL_API_KEY environment variable not set.")
        return None
    quandl.ApiConfig.api_key = api_key

    data_frames = {}
    for ticker in tickers:
        try:
            df = quandl.get(ticker, start_date=start_date, end_date=end_date)
            df.columns = df.columns.str.lower()
            data_frames[ticker] = df
        except Exception as e:
            logger.error(f"Quandl download for {ticker} failed: {e}")
    return data_frames if data_frames else None

def _fetch_alpha_vantage(tickers: list[str], start_date: str, end_date: str, interval: str) -> Optional[dict[str, pd.DataFrame]]:
    """Fetches data from Alpha Vantage, returning a dictionary of DataFrames."""
    from alpha_vantage.timeseries import TimeSeries
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable not set.")
        return None

    ts = TimeSeries(key=api_key, output_format='pandas')
    data_frames = {}
    rename_map = {
        '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
        '5. adjusted close': 'adj close', '6. volume': 'volume', '7. dividend amount': 'dividends',
    }
    for ticker in tickers:
        try:
            data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
            data = data.loc[start_date:end_date]
            data.rename(columns=rename_map, inplace=True)
            final_cols = [col for col in ['open', 'high', 'low', 'close', 'adj close', 'volume', 'dividends'] if col in data.columns]
            data_frames[ticker] = data[final_cols]
        except Exception as e:
            logger.error(f"Could not fetch {ticker} from Alpha Vantage: {e}")
    return data_frames if data_frames else None

def _fetch_fred(tickers: list[str], start_date: str, end_date: str, interval: str) -> Optional[dict[str, pd.DataFrame]]:
    """Fetches data from FRED, returning a dictionary of DataFrames."""
    from fredapi import Fred
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY environment variable not set.")
        return None

    fred = Fred(api_key=api_key)
    data_frames = {}
    for ticker in tickers:
        try:
            series = fred.get_series(ticker, observation_start=start_date, observation_end=end_date)
            df = series.to_frame(name=ticker)
            df.columns = df.columns.str.lower()
            data_frames[ticker] = df
        except Exception as e:
            logger.error(f"Could not fetch {ticker} from FRED: {e}")
    return data_frames if data_frames else None


def fetch_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str,
    source: str
) -> Optional[dict[str, pd.DataFrame]]:
    """
    Fetches financial data from a specified source.
    Returns a dictionary of pandas DataFrames, where each key is a ticker.
    """
    fetcher_map = {
        source.lower(): getattr(sys.modules[__name__], f"_fetch_{source.lower().replace('-', '_')}")
        for source in settings.DATA_SOURCE
        if hasattr(sys.modules[__name__], f"_fetch_{source.lower().replace('-', '_')}")
    }
    fetcher = fetcher_map.get(source.lower())

    if not fetcher:
        logger.error(f"Source '{source}' is not supported.")
        return None

    logger.info(f"Fetching data for {len(tickers)} tickers from {source}...")
    df_dict = fetcher(tickers, start_date, end_date, interval)

    if df_dict:
        logger.info("Data fetched successfully.")
    else:
        logger.warning("Failed to fetch data or no data found.")

    return df_dict

def save_data(
    data_frames: dict[str, pd.DataFrame],
    output_dir: str
) -> None:
    """Saves multiple DataFrames to a specified directory, one CSV per ticker."""
    if not data_frames:
        logger.warning("No data to save.")
        return

    for ticker, df in data_frames.items():
        safe_ticker = ticker.replace('/', '_').replace(':', '_')
        output_path = os.path.join(output_dir, f"{safe_ticker}.csv")
        df.to_csv(output_path)
        logger.info(f"Data for {ticker} saved to {output_path} ({len(df)} rows).")

def main() -> None:
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Fetch financial data and save it as a CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--purpose", type=str, default="invest",
        choices = ["invest", "train"],
        help="Purpose of the data (default: invest)."
    )
    parser.add_argument(
        "tickers", type=str, nargs='+',
        help="List of tickers to fetch (e.g., AAPL MSFT for yfinance, WIKI/AAPL for Quandl)."
    )
    parser.add_argument(
        "--start", type=str, required=True, dest="start_date",
        help="Start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), required=True, dest="end_date",
        help="End date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--interval", type=str, default="1d", required=True, dest="interval",
        help="Interval in 1d format."
    )
    parser.add_argument(
        "--source", type=str, default="yfinance",
        choices=settings.DATA_SOURCE,
        help="Data source to use (default: yfinance)."
    )


    args = parser.parse_args()

    data_dict = fetch_data(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
        interval=args.interval
    )
    output_dir = settings.RAW_DATA_DIR if args.purpose == "train" else settings.TESTING_DATA_DIR
    if data_dict:
        save_data(data_dict, output_dir=str(output_dir))

if __name__ == "__main__":
    main()
