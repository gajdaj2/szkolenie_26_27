from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

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

# Set up the chain
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
chain = prompt | llm | parser

pacjent_1 = """
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
Nie pali papierosów, sporadycznie spożywa alkohol
Wywiad rodzinny: matka chorowała na kamicę żółciową

Badanie fizykalne:

Ciśnienie: 110/70 mmHg, tętno: 95/min, temperatura: 37,8°C
Brzuch: napięty, bolesny w prawym dole biodrowym
Dodatni objaw Blumberga w punkcie McBurneya
Dodatni objaw Rosinga
"""

pacjent_2 = """
Wywiad 2: Pacjent z dusznością
Pacjent: Janusz Nowak, 67 lat, emeryt
Główna skarga: Narastająca duszność wysiłkowa i obrzęki kończyn dolnych
Historia obecnej choroby:

Duszność nasila się od 3 miesięcy
Początkowo występowała przy większym wysiłku, obecnie przy codziennych czynnościach
Musi spać na 2-3 poduszkach (ortopnoe)
Obrzęki kostek i podudzi, szczególnie wieczorem
Sporadyczne kołatania serca
Zmniejszona tolerancja wysiłku - musi odpoczywać po przejściu 50 metrów

Wywiad chorobowy:

Nadciśnienie tętnicze od 15 lat
Przebyty zawał serca 5 lat temu
Cukrzyca typu 2 od 10 lat
Pali papierosy - 1 paczka dziennie przez 40 lat
Leki: amlodypina 10mg, metoprolol 50mg, metformina 1000mg

Wywiad rodzinny:

Ojciec zmarł na zawał serca w wieku 62 lat
Matka chorowała na cukrzycę

Badanie fizykalne:

Ciśnienie: 150/95 mmHg, tętno: 110/min nieregularne
Obrzęki goleni do wysokości kolan
Osłuchowo nad płucami: trzeszczenia w dolnych polach
Serce: arytmiczny rytm, szmer skurczowy 2/6

"""

# Run the chain
output = chain.invoke({"patient_info": pacjent_2})


print("Medical Assessment:")
print(output)
