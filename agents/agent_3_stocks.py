import finnhub
from langchain_core.tools import tool
from dotenv import load_dotenv

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

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(list_of_tools)
query = "Answer in Polish language. What is the current stock price of AAPL and provide recent news about it from 2025-07-23 to 2025-07-23?"

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"get_stock_price": get_stock_price, "get_company_info": get_company_info}[tool_call["name"]]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
final_response = llm_with_tools.invoke(messages)
print(f"Final response: {final_response.content}")