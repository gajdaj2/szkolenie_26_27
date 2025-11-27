from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    Based on the following patient information, provide a medical assessment:
    Patient Information: {patient_info}
    {format_instructions}
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["patient_info"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


def get_tavily_possible_diseases(symptoms: List[str]) -> List[str]:
    """Get possible diseases based on the symptoms using Tavily API."""
    query = " ".join(symptoms)
    response = tavily_client.search(query, limit=5,include_raw_content=True)
    return response

result = get_tavily_possible_diseases(["bóle brzucha", "nudności", "wymioty"])
lista = list(filter(lambda x: x['score'] > 0.7, result['results']))
print(f"Possible diseases based on symptoms: {lista}")
#print(f"Possible diseases based on symptoms: {filter(list(result['results']),lambda x:x['score']>0.8)}")