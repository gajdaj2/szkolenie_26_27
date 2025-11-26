from functools import partial

from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field



class Person(BaseModel):
    first_name: str = Field(description="Imię")
    last_name: str = Field(description="Nazwisko")
    position: str = Field(description="Stanowisko w firmie")

class ManagementBoard(BaseModel):
    members: list[Person] = Field(description="Lista członków zarządu")

load_dotenv()


loader = PyPDFLoader("OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

pages = loader.load()

documents = []

for page in pages:
    doc = Document(page_content=page.page_content, metadata=page.metadata)


print(f"Loaded {len(pages)} pages from the PDF document.")
print(pages[0].page_content)
print(pages[0].metadata)

splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

split_docs = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

vectordb = Chroma.from_documents(documents=split_docs,
                                    embedding=embeddings,
                                 persist_directory="data/chroma_db_orlen"
                                 , collection_name="orlen_hf")

print(vectordb.search("Jaki jest skład zarządu i rady nadzorczej Orlen ?", search_type="similarity", k=2))

template = ("""
Odpowiedz na pytanie na podstawie kontekstu.

Kontekst: {kontekst}

Pytanie: {pytanie}

Zwróć odpowiedź w formacie JSON zgodnym z poniższą specyfikacją:
{format_instructions}

Upewnij się, że odpowiedź jest poprawnym JSON-em.
""")

parser = PydanticOutputParser(pydantic_object=ManagementBoard)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(temperature=0))

prompt = ChatPromptTemplate.from_template(template=template,partial_variables={"format_instructions": parser.get_format_instructions()})

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

retrieval = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})


chain = ({"kontekst": retrieval,"pytanie": RunnablePassthrough()}|
         prompt|
         llm|
         fixing_parser)

output = chain.invoke("Jaka jest karencja na zgon teściowej ?")
print(output)

