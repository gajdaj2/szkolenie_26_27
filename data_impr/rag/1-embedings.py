from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
loader = PyPDFLoader("ORLEN_250821_2025.pdf")

pages = loader.load()

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

documents = []
for page in pages:
    doc = Document(page_content=page.page_content, metadata=page.metadata)
   # doc.metadata["summary"] = llm.invoke(
   #     f"Provide a concise summary of the following text:\n\n{page.page_content}"
   # ).content
    documents.append(doc)

#print(len(documents))
#print(documents[3])


embedding2 = OpenAIEmbeddings(model="text-embedding-ada-002")
#embedding = HuggingFaceBgeEmbeddings(model_name=" ")
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)

#embedding_test = embedding.embed_query("Test embedding")
#print(len(embedding_test))
documents_split = splitter.split_documents(documents)

print(len(documents_split))
print(documents_split[3])


vectordb = Chroma.from_documents(documents=documents_split,
                                 collection_name="owu_bge",
                                 embedding=embedding2,
                                 persist_directory="data/chroma_db_bge")

print(vectordb.search("Jaka jest karencja na zgon członka rodziny?", search_type="similarity", k=2))



