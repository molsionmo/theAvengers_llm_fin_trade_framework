import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaLLM

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent
sys.path.append(str(grandparent_dir))

print(f"添加的路径: {grandparent_dir}")
print(f"路径下是否存在tradingagents: {'tradingagents' in os.listdir(grandparent_dir)}")

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.stockstats_utils import StockstatsUtils


class StrategyAnalyzer:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()

        self.config["llm_provider"] = "ollama"
        self.config["deep_think_llm"] = "llama3.1:8b"
        self.config["quick_think_llm"] = "llama3.1:8b"
        self.config["max_debate_rounds"] = 2
        self.config["online_tools"] = False
        self.config["backend_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.config.pop("openai_api_key", None)
        self.config.pop("openai_base_url", None)
        os.environ.pop("OPENAI_API_KEY", None)

        # 数据配置（仅使用本地数据）
        self.config["data_cache_dir"] = os.path.join(str(current_dir), "data")
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        print(f"数据目录: {self.config['data_cache_dir']}")
        print(f"数据目录中的文件: {os.listdir(self.config['data_cache_dir'])}")

        self._setup_ollama_models()

        self.trading_agent = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=True,
            config=self.config
        )

        self.stockstats_utils = StockstatsUtils()

    def _setup_ollama_models(self):
        def _setup_ollama_models(self):
            self.config["ollama_base_url"] = self.config["backend_url"]
            self.config["chat_model"] = ChatOllama(
                model="llama3.1:8b",
                base_url=self.config["ollama_base_url"],
                temperature=0.7
            )
            self.config["llm"] = OllamaLLM(
                model="llama3.1:8b",
                base_url=self.config["ollama_base_url"],
                temperature=0.7
            )
            self.config["chat_model"] = ChatOllama(
                model="llama3.1:8b",
                base_url=self.config["ollama_base_url"],
                temperature=0.7,
                num_ctx=1024
            )
            self.config["llm"] = OllamaLLM(
                model="llama3.1:8b",
                base_url=self.config["ollama_base_url"],
                temperature=0.7,
                num_ctx=1024
            )

    def get_historical_data_from_local(self, ticker, start_date, end_date):
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        file_path = os.path.join(self.config["data_cache_dir"], filename)

        if not os.path.exists(file_path):
            # 检查是否有其他时间段的同名股票数据
            similar_files = [f for f in os.listdir(self.config["data_cache_dir"])
                             if f.startswith(f"{ticker}_") and f.endswith(".csv")]
            similar_msg = f"\n相似文件: {similar_files}" if similar_files else ""
            raise FileNotFoundError(
                f"本地数据文件不存在: {file_path}{similar_msg}\n"
                f"请确保数据目录下有该股票的CSV文件"
            )

        # 读取CSV文件并尝试识别日期列
        df = pd.read_csv(file_path)

        # 查找可能的日期列（不区分大小写）
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
        if not date_candidates:
            first_col = df.columns[0]
            try:
                # 尝试解析第一列作为日期
                pd.to_datetime(df[first_col])
                date_candidates = [first_col]
            except (ValueError, TypeError):
                raise ValueError(f"CSV文件中未找到日期列，请检查文件格式: {file_path}\n"
                                 f"文件列名: {df.columns.tolist()}")

        # 使用找到的第一个日期列
        date_column = date_candidates[0]
        print(f"使用列 '{date_column}' 作为日期列")

        # 重新读取并解析日期列，确保转换为datetime类型并设置为索引
        df = pd.read_csv(
            file_path,
            parse_dates=[date_column],
            index_col=date_column
        )

        # 灵活识别收盘价列
        close_candidates = [col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()]
        if not close_candidates:
            raise ValueError(f"CSV文件中未找到收盘价列，请检查文件格式: {file_path}\n"
                             f"文件列名: {df.columns.tolist()}")

        # 使用找到的第一个收盘价列
        close_column = close_candidates[0]
        print(f"使用列 '{close_column}' 作为收盘价列")

        # 处理数据
        data = df[[close_column]].rename(columns={close_column: 'price'})
        data['daily_return'] = data['price'].pct_change()

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)

        return data.dropna()

    def get_technical_indicators(self, ticker, curr_date):
        try:
            start_date = self.start_date  # 需要在类中保存
            end_date = self.end_date  # 需要在类中保存

            rsi = self.stockstats_utils.get_stock_stats(
                symbol=ticker,
                indicator='rsi_14',
                curr_date=curr_date,
                start_date=start_date,
                end_date=end_date,
                data_dir=self.config["data_cache_dir"],
                online=self.config["online_tools"]
            )
            macd = self.stockstats_utils.get_stock_stats(
                symbol=ticker,
                indicator='macd',
                curr_date=curr_date,
                start_date=start_date,
                end_date=end_date,
                data_dir=self.config["data_cache_dir"],
                online=self.config["online_tools"]
            )
            return {"rsi": rsi, "macd": macd}
        except Exception as e:
            print(f"技术指标获取失败: {e}")
            return {"rsi": None, "macd": None}

            # 读取本地Alpha Vantage数据
            filename = f"{ticker}_{self.start_date}_{self.end_date}.csv"
            file_path = os.path.join(self.config["data_cache_dir"], filename)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # 计算RSI（14天）
            delta = df['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # 计算MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26

            # 获取当前日期的指标值
            curr_rsi = rsi.loc[curr_date] if curr_date in rsi.index else None
            curr_macd = macd.loc[curr_date] if curr_date in macd.index else None

            return {"rsi": curr_rsi, "macd": curr_macd}
        except Exception as e:
            print(f"技术指标计算失败: {e}")
            return {"rsi": None, "macd": None}

    def generate_daily_decision(self, ticker, curr_date):
        """基于仓库多代理框架生成每日交易决策"""
        try:
            full_state, decision = self.trading_agent.propagate(
                company_name=ticker,
                trade_date=curr_date
            )
            return decision.strip().upper()
        except Exception as e:
            print(f"决策生成失败（{curr_date}）: {e}")
            return "HOLD"

    def simulate_strategy(self, ticker, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        price_data = self.get_historical_data_from_local(ticker, start_date, end_date)
        """回测策略：仅使用本地数据计算回报率"""
        price_data = self.get_historical_data_from_local(ticker, start_date, end_date)
        if price_data.empty:
            raise ValueError("没有获取到有效的价格数据")

        price_data['position'] = 0
        price_data['strategy_return'] = 0.0
        price_data['decision'] = "HOLD"

        for date in price_data.index[1:]:
            curr_date = date.strftime("%Y-%m-%d")
            prev_date = price_data.index[price_data.index.get_loc(date) - 1]

            _ = self.get_technical_indicators(ticker, curr_date)
            decision = self.generate_daily_decision(ticker, curr_date)
            price_data.at[date, 'decision'] = decision

            prev_position = price_data.at[prev_date, 'position']
            if decision == "BUY" and prev_position == 0:
                price_data.at[date, 'position'] = 1
            elif decision == "SELL" and prev_position == 1:
                price_data.at[date, 'position'] = 0
            else:
                price_data.at[date, 'position'] = prev_position

            price_data.at[date, 'strategy_return'] = (
                    price_data.at[date, 'daily_return'] * price_data.at[date, 'position']
            )

        price_data['benchmark_cumulative'] = (1 + price_data['daily_return']).cumprod() - 1
        price_data['strategy_cumulative'] = (1 + price_data['strategy_return']).cumprod() - 1

        return price_data

    def plot_results(self, results, ticker):
        """生成累计回报率对比折线图"""
        plt.figure(figsize=(12, 6))
        plt.plot(
            results.index,
            results['benchmark_cumulative'] * 100,
            label='基准（买入持有）',
            color='blue',
            alpha=0.7
        )
        plt.plot(
            results.index,
            results['strategy_cumulative'] * 100,
            label='TradingAgents策略',
            color='red',
            alpha=0.7
        )

        buy_signals = results[results['decision'] == "BUY"]
        sell_signals = results[results['decision'] == "SELL"]
        plt.scatter(buy_signals.index, buy_signals['strategy_cumulative'] * 100,
                    marker='^', color='g', label='买入', zorder=3)
        plt.scatter(sell_signals.index, sell_signals['strategy_cumulative'] * 100,
                    marker='v', color='k', label='卖出', zorder=3)

        plt.title(f'{ticker} 累计回报率对比（{results.index[0].date()} 至 {results.index[-1].date()}）', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累计回报率（%）', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_path = f'{ticker}_strategy_vs_benchmark.png'
        plt.savefig(output_path, dpi=300)
        print(f"折线图已保存至: {output_path}")
        plt.show()

    def run(self, ticker, start_date, end_date):
        """运行完整分析流程"""
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False

        print(f"开始分析 {ticker}（{start_date} 至 {end_date}）...")
        results = self.simulate_strategy(ticker, start_date, end_date)

        final_benchmark = results['benchmark_cumulative'].iloc[-1] * 100
        final_strategy = results['strategy_cumulative'].iloc[-1] * 100
        print(f"\n分析结束：")
        print(f"基准（买入持有）最终回报率: {final_benchmark:.2f}%")
        print(f"TradingAgents策略最终回报率: {final_strategy:.2f}%")
        print(f"超额收益: {final_strategy - final_benchmark:.2f}%")

        self.plot_results(results, ticker)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用TradingAgents框架进行策略回测（仅本地数据）')
    parser.add_argument('--ticker', required=True, help='股票代码（如AAPL）')
    parser.add_argument('--start-date', required=True, help='开始日期（YYYY-MM-DD）')
    parser.add_argument('--end-date', required=True, help='结束日期（YYYY-MM-DD）')
    args = parser.parse_args()

    analyzer = StrategyAnalyzer()
    analyzer.run(args.ticker, args.start_date, args.end_date)