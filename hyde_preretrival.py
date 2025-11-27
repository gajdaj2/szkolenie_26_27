from dotenv import load_dotenv
from langchain.embeddings import HypotheticalDocumentEmbedder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

load_dotenv()


loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1024,chunk_overlap=60)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())

embeddings = HypotheticalDocumentEmbedder.from_llm(OpenAI(n=3,best_of=4),OpenAIEmbeddings(),"web_search")

query = "Jaka jest karencja na zgon członka rodziny?"

result = embeddings.embed_query(query)
print(result)

vectorstore.similarity_search(query)

system = """
As a knowledgeable and helpful research assistant, your task is to provide informative answers based on the given context.
Use your extensive knowledge base to offer clear, concise, and accurate responses to the user's inquiries.
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
context = prompt | llm | StrOutputParser()


answer = context.invoke(
    {
        "jak jest karencja na zgon członka rodziny ?"
    }
)

print(answer)

chain = RunnablePassthrough.assign(hypothetical_document=context)

out = chain.invoke(
    {
        "question": "Jak jest karencja na zgon członka rodziny ?"
    }
)

print(out)

output = vectorstore.similarity_search(out['hypothetical_document'])
print(output)