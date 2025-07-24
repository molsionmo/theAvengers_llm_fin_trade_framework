import os
import sys
import time
import random
import argparse
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta


def get_stock_prices(ticker, start_date, end_date, debug=False):
    try:
        if debug:
            print("调试模式已启用")
            print(f"获取 {ticker} 从 {start_date} 到 {end_date} 的数据...")

        # 增加随机延迟，避免请求过于频繁
        delay = random.uniform(1, 3)
        time.sleep(delay)

        # 替换为你的 Alpha Vantage API 密钥
        api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # 请替换为实际API密钥
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')

        # 转换索引为 datetime 格式
        data.index = pd.to_datetime(data.index)

        # 生成完整的日期范围（包括所有日期，无论是否为交易日）
        full_date_range = pd.date_range(start=start_date, end=end_date)

        # 筛选出指定日期范围内的数据
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data.loc[mask].copy()

        # 重命名列
        filtered_data = filtered_data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })

        # 重新索引以包含所有日期（包括非交易日）
        full_data = filtered_data.reindex(full_date_range)

        # 处理缺失值（非交易日）
        # 填充价格数据（使用前一个交易日的值）
        price_columns = ['Open', 'High', 'Low', 'Close']
        full_data[price_columns] = full_data[price_columns].fillna(method='ffill')

        # 成交量在非交易日填充为0
        full_data['Volume'] = full_data['Volume'].fillna(0)

        # 计算涨跌幅（非交易日涨跌幅为0）
        full_data['Change %'] = full_data['Close'].pct_change() * 100
        full_data['Change %'] = full_data['Change %'].fillna(0)

        # 计算其他技术指标
        full_data['Average Price'] = (full_data['High'] + full_data['Low']) / 2
        full_data['Previous Close'] = full_data['Close'].shift(1).fillna(method='ffill')

        # 计算52周最高价和最低价
        full_data['52 Week High'] = full_data['High'].rolling(window=252, min_periods=1).max()
        full_data['52 Week Low'] = full_data['Low'].rolling(window=252, min_periods=1).min()

        # 计算成交额和加权平均价格
        full_data['Turnover'] = full_data['Volume'] * full_data['Close']
        full_data['VWAP'] = (full_data['Volume'] * full_data['Average Price']).cumsum() / full_data[
            'Volume'].cumsum().replace(0, 1)

        # 确保索引格式为日期字符串
        full_data.index = full_data.index.strftime('%Y-%m-%d')

        if debug:
            print(f"数据处理完成，包含 {len(full_data)} 天（包括非交易日）")

        return full_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Change %',
                          'Average Price', 'Previous Close', '52 Week High',
                          '52 Week Low', 'Turnover', 'VWAP']]

    except Exception as e:
        print(f"获取数据时出现错误: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="获取指定公司股票数据并存储在data文件夹")
    parser.add_argument("--ticker", type=str, required=True, help="公司的股票代码")
    parser.add_argument("--start-date", type=str, required=True, help="开始日期，格式为 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="结束日期，格式为 YYYY-MM-DD")
    parser.add_argument("--output", type=str, help="输出文件名称，格式为 CSV（将保存到data目录）")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date
    output_filename = args.output
    debug = args.debug

    # 验证日期格式
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("错误：日期格式必须为 YYYY-MM-DD")
        sys.exit(1)

    # 创建 data 目录（如果不存在）
    output_dir = "data"  # 直接使用data文件夹，而非training/data
    os.makedirs(output_dir, exist_ok=True)

    # 确定输出文件路径
    if output_filename:
        output_file = os.path.join(output_dir, output_filename)
    else:
        # 默认文件名格式：股票代码_起始日期_结束日期.csv
        output_file = os.path.join(output_dir, f"{ticker}_{start_date}_{end_date}.csv")

    # 获取股票数据（包含完整日期范围）
    prices = get_stock_prices(ticker, start_date, end_date, debug)

    if prices is not None:
        # 保存数据到 CSV
        prices.to_csv(output_file)
        print(f"数据已保存到 {output_file}")
        print(f"日期范围：{start_date} 至 {end_date}（共 {len(prices)} 天）")
    else:
        print(f"无法获取 {ticker} 的数据")
