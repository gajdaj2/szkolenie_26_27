from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Definicja narzędzia do dodawania z jasnym opisem parametrów
@tool
def add(x: float, y: float) -> float:
    """Add two numbers x and y together.

    Args:
        x: First number to add
        y: Second number to add

    Returns:
        The sum of x and y
    """
    result = x + y
    print(f"Adding {x} + {y} = {result}")
    return result

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers x and y together.

    Args:
        x: First number to multiply
        y: Second number to multiply

    Returns:
        The product of x and y
    """
    result = x * y
    print(f"Multiplying {x} * {y} = {result}")
    return result

@tool
def subtract(x: float, y: float) -> float:
    """Subtract y from x.

    Args:
        x: Number to subtract from
        y: Number to subtract

    Returns:
        The difference x - y
    """
    result = x - y
    print(f"Subtracting {x} - {y} = {result}")
    return result

# Lista narzędzi
tools = [add, multiply, subtract]

# Konfiguracja LLM z narzędziami
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Funkcja do wywołania agenta
def run_agent(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    messages = [HumanMessage(content=query)]

    # Pierwszy krok - LLM decyduje jakie narzędzia użyć
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Wykonaj wywołania narzędzi
    if ai_msg.tool_calls:
        print(f"\nAgent decided to use {len(ai_msg.tool_calls)} tool(s):")

        for tool_call in ai_msg.tool_calls:
            print(f"\nTool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

            # Wybierz odpowiednie narzędzie
            selected_tool = {
                "add": add,
                "multiply": multiply,
                "subtract": subtract
            }[tool_call["name"]]

            # Wywołaj narzędzie
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)

    # Ostateczna odpowiedź
    final_response = llm_with_tools.invoke(messages)

    print(f"\n{'='*60}")
    print(f"Final Answer: {final_response.content}")
    print('='*60)

    return final_response.content


