from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
import os

from dotenv import load_dotenv

load_dotenv()

from phi.model.azure import AzureOpenAIChat

azure_model = AzureOpenAIChat(
    id=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)


finance_agent = Agent(
    name="Finance Agent",
    model=azure_model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
finance_agent.print_response("Which stock is a better investment, NVDA or META?", stream=True)
