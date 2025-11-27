from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
loaders = [
PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf"),
PyPDFLoader("OWU_Warta_MotoAssistance_C4718_IPID.pdf"),
]

docs = []
for doc in loaders:
    docs.extend(doc.load())

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400,chunk_overlap=60)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings(), persist_directory="chroma_db_multiquery")

retriever = vectorstore.as_retriever()

template = """
You are an AI language model assistant tasked with generating informative queries for a vector search engine.
The user has a question: "{question}"
Your goal is to create three variations of this question that capture different aspects of the user's intent. These variations will help the search engine retrieve relevant documents even if they don't use the exact keywords as the original question.
Provide these alternative questions, each on a new line.**
Original question: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (prompt_perspectives
                    |ChatOpenAI(temperature=0)
                    |StrOutputParser()
                    |(lambda x: x.split("\n"))
                    )


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs z ich zawartością"""
    flattened_docs = [doc for sublist in documents for doc in sublist]

    # Usuwamy duplikaty na podstawie zawartości
    seen_content = set()
    unique_docs = []

    for doc in flattened_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

    # Zwracamy zawartość dokumentów jako string
    return "\n\n".join(doc.page_content for doc in unique_docs)


# Retrieve
question = "Jaka jest karencja na zgon członka rodziny?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
print(len(docs))


# RAG
template = """Answer the following question based on this context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": lambda x: get_unique_union(retriever.map().invoke(generate_queries.invoke(x))),
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)
output = final_rag_chain.invoke({"question":question})
print(output)