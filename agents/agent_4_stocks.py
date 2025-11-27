import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import finnhub
load_dotenv()


finnhub_client = finnhub.Client(api_key="d20ht6hr01qrk6cl3a70d20ht6hr01qrk6cl3a7g")


@tool
def get_stock_price(symbol: str) -> float:
    """Get the current stock price for the given symbol."""
    quote = finnhub_client.quote(symbol)
    return quote['c']


@tool
def get_company_info(symbol: str, from_date: str, to_date: str) -> dict:
    """Get company information for the given symbol. and date range. 
    from_date and to_date should be in 'YYYY-MM-DD' format."""
    company_info = finnhub_client.company_news(symbol=symbol, _from=from_date, to=to_date)
    list_of_news = []
    for news in company_info:
        news_item = {
            "headline": news['headline'],
            "summary": news['summary'],
            "source": news['source'],
        }
        list_of_news.append(news_item)
    return list_of_news


list_of_tools = [get_stock_price, get_company_info]

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)


system_prompt = """You are stock market helpful assistant that provides stock information and news in Polish.
You will be provided with tools to get stock prices and company news.
Create concise and informative responses based on the tools provided.
Create reports in Polish language.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}"),
    MessagesPlaceholder("agent_scratchpad")
])


agent = create_openai_functions_agent(
    llm=llm,
    tools=list_of_tools,
    prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=list_of_tools,
    verbose=True,
    return_intermediate_steps=True
)

result = agent_executor.invoke({
    "query": """Answer in Polish language. What is the current stock price of ORLEN 
    and provide recent news about it from 2025-07-19 to 2025-07-23?"""
})

print(f"Final response: {result['output']}")
