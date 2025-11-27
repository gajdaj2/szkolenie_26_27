from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tavily import TavilyClient


load_dotenv()

# Define the patient data structure
class PatientAssessment(BaseModel):
    diagnosis: str = Field(description="Primary medical diagnosis")
    pain_level: int = Field(description="Pain level on scale of 0-10")
    symptoms: List[str] = Field(description="List of reported symptoms")
    requires_hospitalization: bool = Field(
        description="Whether the patient needs to be hospitalized")
    possible_diseases: List[str] = Field(
        description="List of possible diseases based on the symptoms"
    )

# Create the parser
parser = PydanticOutputParser(pydantic_object=PatientAssessment)
tavily_client = TavilyClient(api_key="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")

# Create the prompt template
template = """
You are a helpful medical assistant. You MUST respond using ONLY ENGLISH keywords for actions (Thought, Action, Action Input, Observation, Final Answer), but you can provide the final medical assessment in Polish.

IMPORTANT: You MUST use these exact English words: "Thought:", "Action:", "Action Input:", "Observation:", "Final Answer:"
DO NOT translate these keywords to Polish (do NOT use "Myśl", "Działanie", etc.)

You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do (in English)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final medical assessment (you can write this in Polish)

Begin!

Patient Information: {patient_info}

Thought: {agent_scratchpad}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["patient_info", "tools", "tool_names", "agent_scratchpad"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


@tool
def get_tavily_possible_diseases(symptoms: str) -> dict:
    """Get possible diseases based on the symptoms using Tavily API.

    Args:
        symptoms: A comma-separated string of symptoms or a description of symptoms
    """
    # Jeśli symptoms to string, używamy go bezpośrednio jako query
    query = symptoms
    response = tavily_client.search(query, limit=5, include_raw_content=True)
    return response

# Użyj .invoke() zamiast bezpośredniego wywołania i przekaż słownik
result = get_tavily_possible_diseases.invoke({"symptoms": "bóle brzucha, nudności, wymioty"})

# Sprawdź czy result jest słownikiem i ma klucz 'results'
if isinstance(result, dict) and 'results' in result:
    lista = [item for item in result['results'] if item.get('score', 0) > 0.7]
    print(f"Possible diseases based on symptoms: {lista}")
else:
    print(f"Unexpected response format: {result}")

# Set up the chain
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

pacjent_1 = """ Odpowiedz w języku polskim.
Wywiad 1: Pacjentka z bólem brzucha
Pacjentka: Anna Kowalska, 34 lata, nauczycielka
Główna skarga: Ostry ból w prawym podbrzuszu trwający od 6 godzin
Historia obecnej choroby:
Ból rozpoczął się nagle dziś rano około godziny 6:00
Początkowo zlokalizowany w okolicy pępka, następnie przeniósł się do prawego dołu biodrowego
Charakter bólu: ostry, narastający, stały
Nasilenie: 8/10 w skali bólu
Towarzyszące objawy: nudności, jednorazowe wymioty, brak apetytu
Pacjentka przyjęła 2 tabletki ibuprofenu bez znaczącej poprawy
Wywiad chorobowy:
Ostatnia miesiączka: 2 tygodnie temu, regularna
Nie przyjmuje leków na stałe
"""

agent = create_react_agent(llm, [get_tavily_possible_diseases], prompt)
agent_executor2 = AgentExecutor(
    agent=agent,
    tools=[get_tavily_possible_diseases],
    verbose=True,
    handle_parsing_errors=True
)
result = agent_executor2.invoke({"patient_info": pacjent_1})
print(f"Final response: {result['output']}")