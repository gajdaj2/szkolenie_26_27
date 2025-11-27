from langchain_community.document_loaders import PyPDFLoader


from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

chunk_size = 1024
chunk_overlap = 20

separators = ["\n\n", "\n"]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

documents = loader.load()

docs = text_splitter.split_documents(documents)

for doc in docs:
    print(doc.page_content)
    print("-----")


embeddings = OpenAIEmbeddings()

db_chroma = Chroma.from_documents(docs, embeddings,collection_name="OWU", persist_directory="chroma_db_recursive")

result = db_chroma.similarity_search(query="Jaka jest karencja na zgon członka rodziny?", k=2)

print(result)