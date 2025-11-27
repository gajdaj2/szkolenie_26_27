from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


load_dotenv()


loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

docs = loader.load()


breakpoint_threshold_type="percentile"

text_splitter = SemanticChunker(embeddings=OpenAIEmbeddings(), breakpoint_threshold_type=breakpoint_threshold_type)

documents = text_splitter.split_documents(docs)

print(documents)

db_chroma = Chroma.from_documents(documents=documents,embedding=OpenAIEmbeddings(), persist_directory="chroma_db_semantic")

res = db_chroma.similarity_search(query="Jaka jest karencja na zgon członka rodziny?", k=2)

for result in res:
    print(result.page_content)
    print("-----")
