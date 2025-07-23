from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule
import questionary
import pandas as pd
import yfinance as yf
import json
import sys
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi - Agents LLM Financial Trading Framework",
    add_completion=True
)


class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None
        self.agent_status = {
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            "Fundamentals Analyst": "pending",
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            "Trader": "pending",
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            "Portfolio Manager": "pending"
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        latest_section = None
        latest_content = None
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision"
            }
            self.current_report = f"### {section_titles[latest_section]}\n{latest_content}"
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []
        if any(self.report_sections[section] for section in [
            "market_report", "sentiment_report", "news_report", "fundamentals_report"
        ]):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["market_report"]:
                report_parts.append(f"### Market Analysis\n{self.report_sections['market_report']}")
            if self.report_sections["sentiment_report"]:
                report_parts.append(f"### Social Sentiment\n{self.report_sections['sentiment_report']}")
            if self.report_sections["news_report"]:
                report_parts.append(f"### News Analysis\n{self.report_sections['news_report']}")
            if self.report_sections["fundamentals_report"]:
                report_parts.append(f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}")

        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True
        )
    )

    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,
        padding=(0, 2),
        expand=True
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst"
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"]
    }

    for team, agents in teams.items():
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red"
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red"
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )


def get_user_input():
    options = ["Analyze Ticker", "Exit"]
    choice = questionary.select(
        "What would you like to do?",
        choices=options
    ).ask()

    if choice == "Exit":
        sys.exit(0)

    ticker = questionary.text(
        "Enter the ticker symbol to analyze:",
        validate=lambda x: len(x.strip()) > 0 or "Please enter a valid ticker symbol."
    ).ask()
    start_date = questionary.text(
        "Enter the start date (YYYY-MM-DD):",
        validate=lambda x: len(x.strip()) > 0 and datetime.datetime.strptime(x.strip(), "%Y-%m-%d") or "Please enter a valid date in YYYY-MM-DD format."
    ).ask()
    end_date = questionary.text(
        "Enter the end date (YYYY-MM-DD):",
        validate=lambda x: len(x.strip()) > 0 and datetime.datetime.strptime(x.strip(), "%Y-%m-%d") or "Please enter a valid date in YYYY-MM-DD format."
    ).ask()
    return ticker, start_date, end_date


def calculate_cumulative_returns(data):
    returns = data['Close'].pct_change()
    cumulative_returns = (1 + returns).cumprod()
    return cumulative_returns


def calculate_annualized_returns(data):
    daily_returns = data['Close'].pct_change()
    annualized_returns = (1 + daily_returns.mean()) ** 252 - 1
    return annualized_returns


def calculate_sharpe_ratio(data):
    daily_returns = data['Close'].pct_change()
    risk_free_rate = 0.0
    sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * (252 ** 0.5)
    return sharpe_ratio


def calculate_max_drawdown(data):
    cumulative_returns = calculate_cumulative_returns(data)
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown


@app.command()
def main():
    layout = create_layout()
    with Live(layout, refresh_per_second=1):
        ticker, start_date, end_date = get_user_input()
        data = yf.download(ticker, start=start_date, end=end_date)

        cumulative_returns = calculate_cumulative_returns(data)
        annualized_returns = calculate_annualized_returns(data)
        sharpe_ratio = calculate_sharpe_ratio(data)
        max_drawdown = calculate_max_drawdown(data)

        folder_name = f"{ticker}_{start_date}_{end_date}"
        import os
        os.makedirs(folder_name, exist_ok=True)

        returns_df = pd.DataFrame(cumulative_returns, columns=['Cumulative Returns'])
        returns_df.to_csv(os.path.join(folder_name, 'returns.csv'))

        metrics = {
            "Annualized Returns": annualized_returns,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }
        with open(os.path.join(folder_name, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Data saved to {folder_name}")

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "google"
        config["backend_url"] = "https://generativelanguage.googleapis.com/v1"
        config["deep_think_llm"] = "gemini-2.0-flash"
        config["quick_think_llm"] = "gemini-2.0-flash"
        config["max_debate_rounds"] = 1
        config["online_tools"] = True

        ta = TradingAgentsGraph(debug=True, config=config)
        _, decision = ta.propagate(ticker, end_date)
        print(decision)


if __name__ == "__main__":
    app()