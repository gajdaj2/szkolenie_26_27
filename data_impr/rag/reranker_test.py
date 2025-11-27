import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import vector_stores

load_dotenv()

# loader = PyPDFLoader("OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")
#
# pages = loader.load()
#
# print(f"Loaded {len(pages)} pages from the PDF document.")
#
# splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=60)
#
# docs = splitter.split_documents(pages)
#
# print(f"Split into {len(docs)} chunks.")
#
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# vector_store = Chroma.from_documents(documents=docs,
#                                      collection_name="owu",
#                                      embedding=embeddings,
#                                      persist_directory="data/chroma_db2")


client = chromadb.PersistentClient(path="data/chroma_db2")
vector_store = Chroma(client=client,collection_name="owu",embedding_function=OpenAIEmbeddings())
query = "Jaka jest karencja na zgon członka rodziny?"

results = vector_store.search(query,search_type="similarity", k=3)
print([result.id for result in results])
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

print(compression_retriever.invoke("Jaka jest karencja na zgon członka rodziny?"))


chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

print(chain.invoke("Na jaki okres przysługuje szpital ?")['result'])








