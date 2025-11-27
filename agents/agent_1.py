from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
# Define tools using concise function definitions
@tool
def multiply(x: float, y: float) -> float:
   """Multiply 'x' times 'y'."""
   return x * y
 
@tool
def exponentiate(x: float, y: float) -> float:
   """Raise 'x' to the 'y'."""
   return x**y
 
@tool
def add(x: float, y: float) -> float:
   """Add 'x' and 'y'."""
   return x + y


tools = [multiply, exponentiate, add]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)


query = "What is 2 raised to the power of 3, multiplied by 4, and then added to 5?"

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
   selected_tool = {"add": add, "multiply": multiply, "exponentiate": exponentiate}[tool_call["name"]]
   tool_msg = selected_tool.invoke(tool_call)
   print(f"{tool_msg}")   
   messages.append(tool_msg)

final_response = llm_with_tools.invoke(messages)
print(f"Final response: {final_response.content}")




