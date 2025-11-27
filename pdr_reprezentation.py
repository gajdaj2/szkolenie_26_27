from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter



load_dotenv()


embedding = OpenAIEmbeddings()

loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

docs = loader.load()

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
store = InMemoryStore()



vectorstore = Chroma(collection_name="full_documents", embedding_function=embedding)

full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter
)


full_doc_retriever.add_documents(docs)

print(list(store.yield_keys()))

sub_docs = vectorstore.similarity_search("Jaka jest karencja na zgon członka rodziny?", k=2)
print(len(sub_docs))

print(sub_docs[1].page_content)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Adding documents
big_chunks_retriever.add_documents(docs)
print(len(list(store.yield_keys())))  #

sub_docs = vectorstore.similarity_search("Jaka jest karencja na zgon członka rodziny", k=2)
print(len(sub_docs))

print(sub_docs[0].page_content)

retrieved_docs = big_chunks_retriever.invoke("Jaka jest karencja na zgon członka rodziny?")
print(len(retrieved_docs))

print(len(retrieved_docs[0].page_content))
print(retrieved_docs[0].page_content)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model="gpt-4o-mini", temperature=0.5, verbose=True),
    chain_type="stuff",
    retriever=big_chunks_retriever,
)

query = "Jaka jest karencja na zgon członka rodziny?"
response = qa.invoke(query)
print(response)




