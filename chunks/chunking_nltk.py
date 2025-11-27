import nltk
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter

nltk.download('punkt',quiet=True)

loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

documents = loader.load()

text_splitter = NLTKTextSplitter()

texts = text_splitter.split_documents(documents)
for sentence in texts:
    print(sentence.page_content)
    print("-----")


