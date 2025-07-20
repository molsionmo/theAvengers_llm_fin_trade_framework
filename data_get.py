import os
import sys
import time
import random
import argparse
import pandas as pd
from pandas_datareader import data as pdr
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import questionary

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
}


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "Enter the analysis date (YYYY-MM-DD):",
        validate=lambda x: validate_date(x.strip())
        or "Please enter a valid date in YYYY-MM-DD format.",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        print("\n[red]No date provided. Exiting...[/red]")
        exit(1)

    return date.strip()


# 将 TradingAgents 所在的父目录添加到 sys.path 中
sys.path.append('D:/TradingAgents')


def get_stock_prices(ticker, start_date, end_date, debug=False):
    try:
        if debug:
            print("调试模式已启用")
            print("使用 Alpha Vantage 数据获取股票价格...")

        # 增加随机延迟，范围在 1 到 3 秒之间，降低请求频率
        delay = random.uniform(1, 3)
        time.sleep(delay)

        # 替换为你自己的 Alpha Vantage API 密钥
        api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        data = data[(data.index >= start_date) & (data.index <= end_date)]

        if data.empty:
            print(f"警告: 没有找到 {ticker} 在 {start_date} 到 {end_date} 的数据")
            return None
        else:
            # 重命名列
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })

            # 计算涨跌幅
            data['Change %'] = data['Close'].pct_change() * 100

            # 计算均价
            data['Average Price'] = (data['High'] + data['Low']) / 2

            # 计算前收盘价
            data['Previous Close'] = data['Close'].shift(1)

            # 计算52周最高价和最低价
            data['52 Week High'] = data['High'].rolling(window=52 * 5, min_periods=1).max()
            data['52 Week Low'] = data['Low'].rolling(window=52 * 5, min_periods=1).min()

            # 计算成交额
            data['Turnover'] = data['Volume'] * data['Close']

            # 计算加权平均价格
            data['VWAP'] = (data['Volume'] * data['Average Price']).cumsum() / data['Volume'].cumsum()

            return data[['Open', 'High', 'Low', 'Close', 'Volume', 'Change %', 'Average Price', 'Previous Close',
                         '52 Week High', '52 Week Low', 'Turnover', 'VWAP']]
    except Exception as e:
        print(f"获取数据时出现错误: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="获取指定公司在限定日期内的开盘价和收盘价")
    parser.add_argument("--ticker", type=str, required=True, help="公司的股票代码")
    parser.add_argument("--start-date", type=str, required=True, help="开始日期，格式为 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="结束日期，格式为 YYYY-MM-DD")
    parser.add_argument("--output", type=str, help="输出文件路径，格式为 CSV")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date
    output_file = args.output
    debug = args.debug

    prices = get_stock_prices(ticker, start_date, end_date, debug)

    if prices is not None:
        print(f"{ticker} 在 {start_date} 到 {end_date} 的相关数据:")
        print(prices)

        if output_file:
            prices.to_csv(output_file)
            print(f"数据已保存到 {output_file}")

prices = get_stock_prices(ticker, start_date, end_date, debug)
if prices is not None:
    print(f"成功获取 {ticker} 在 {start_date} 到 {end_date} 的数据，数据行数: {len(prices)}")
else:
     print(f"未获取到 {ticker} 在 {start_date} 到 {end_date} 的数据")