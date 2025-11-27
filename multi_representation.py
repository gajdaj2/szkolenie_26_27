import uuid

from dotenv import load_dotenv
from langchain.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.stores import InMemoryByteStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

documents = loader.load()

print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(documents)

chain = (
    {"doc":lambda x:x.page_content}|
    ChatPromptTemplate.from_template("Summarize the following document:\n{doc}")|
    ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)|
    StrOutputParser()
)

summaries = chain.batch(docs,max_concurrency=3)

vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings(),persist_directory="chroma_db_summaries_2")

retriver = vectorstore.as_retriever()

store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

query = "Jaka jest karencja na zgon członka rodziny?"
sub_docs = vectorstore.similarity_search(query)
sub_docs[0]

retrieved_docs = retriever.invoke(query)

text = "\n\n".join([doc.page_content for doc in retrieved_docs])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, verbose=True)

prompt = ChatPromptTemplate.from_template(
    "Bazując na poniższych informacjach, odpowiedz na pytanie: {query}. \n\n{text}"
)
query = "Jaka jest karencja na zgon członka rodziny?"

# print(query)
# print(text)
# response = llm.invoke(f"Bazując na poniższych informacjach, odpowiedz na pytanie: {query}. \n\n{text}")
# response2 = llm.invoke(prompt.format(query=query, text=text))
# print(response)


rag_chain = (
    {"text": retriever, "query": RunnablePassthrough()} |
    prompt |
    llm |
    StrOutputParser()
)

output = rag_chain.invoke(query)
print(output)