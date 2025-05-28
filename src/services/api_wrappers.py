from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
import os

def get_alpha_vantage() -> AlphaVantageAPIWrapper:
    """Configure AlphaVantage API wrapper"""
    return AlphaVantageAPIWrapper(
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY")
    )

def get_yahoo_finance() -> YahooFinanceNewsTool:
    """Configure Yahoo Finance tool"""
    return YahooFinanceNewsTool()