from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Inicjalizacja modelu
llm = ChatOllama(model="llama3.2", temperature=0.7)
parser = StrOutputParser()

# Definicja promptów
przepis_prompt = ChatPromptTemplate.from_messages([
    ("system", "Jesteś szefem kuchni. Twórz przepisy w języku polskim."),
    ("human", "Stwórz przepis na: {danie}")
])

gramatura_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analizuj gramaturę składników w przepisach."),
    ("human", "Przepis:\n{przepis}\n\nPodaj gramaturę składników:")
])

czas_prompt = ChatPromptTemplate.from_messages([
    ("system", "Szacuj czas przygotowania potraw."),
    ("human", "Przepis:\n{przepis}\n\nOkreśl czas przygotowania:")
])

# Budowanie sekwencyjnego łańcucha
full_chain = (
    {"danie": RunnablePassthrough()}
    | przepis_prompt
    | llm
    | parser
    | (lambda przepis: {
        "przepis": przepis,
        "gramatura": (gramatura_prompt | llm | parser).invoke({"przepis": przepis}),
        "czas": (czas_prompt | llm | parser).invoke({"przepis": przepis})
    })
)

# Użycie
result = full_chain.invoke("spaghetti bolognese")
print("PRZEPIS:\n", result["przepis"])
print("\nGRAMATURA:\n", result["gramatura"])
print("\nCZAS:\n", result["czas"])