from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

documents = loader.load()

print(len(documents))
print(documents[0].page_content)


chunk_size = 256
chunk_overlap = 20

text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

all_chunks = []

for pages in documents:
    chunks = text_splitter.split_text(pages.page_content)
    all_chunks.extend(chunks)

for docs in all_chunks:
    print(docs)


embeddings = OpenAIEmbeddings()

db_chroma = Chroma.from_texts(all_chunks, embeddings, persist_directory="chroma_db3")

result = db_chroma.similarity_search(query="Jaka jest karencja na zgon członka rodziny?", k=2)

print(result)




