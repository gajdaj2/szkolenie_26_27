from dotenv import load_dotenv

load_dotenv()


import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


loaders = [
    PyPDFLoader('OWU_Warta_MotoAssistance_C4718_IPID.pdf'),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=60)
splits = text_splitter.split_documents(docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

query = "Jakie są korzyści z posiadania ubezpieczenia Warta MotoAssistance?"
docs = retriever.invoke(query)
print(docs)


model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

compressor = CrossEncoderReranker(model=model, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke("Jakie są korzyści z posiadania ubezpieczenia Warta MotoAssistance?")

for doc in compressed_docs:
    print(doc.page_content)


### Print the compressed documents


print(compressed_docs)