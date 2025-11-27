from dotenv import load_dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, OpenAI



load_dotenv()


docs = [
    Document(
        page_content="A complex, layered narrative exploring themes of identity and belonging",
        metadata={"title":"The Namesake", "author": "Jhumpa Lahiri", "year": 2003, "genre": "Fiction", "rating": 4.5, "language":"English", "country":"USA"},
    ),
    Document(
        page_content="A luxurious, heartfelt novel with themes of love and loss set against a historical backdrop",
        metadata={"title":"The Nightingale", "author": "Kristin Hannah", "year": 2015, "genre": "Historical Fiction", "rating": 4.8, "language":"English", "country":"France"},
    ),
    Document(
        page_content="A full-bodied epic with rich characters and a sprawling plot",
        metadata={"title":"War and Peace", "author": "Leo Tolstoy", "year": 1869, "genre": "Historical Fiction", "rating": 4.7, "language":"Russian", "country":"Russia"},
    ),
    Document(
        page_content="An elegant, balanced narrative with intricate character development and subtle themes",
        metadata={"title":"Pride and Prejudice", "author": "Jane Austen", "year": 1813, "genre": "Romance", "rating": 4.6, "language":"English", "country":"UK"},
    ),
    Document(
        page_content="A highly regarded novel with deep themes and a nuanced exploration of human nature",
        metadata={"title":"To Kill a Mockingbird", "author": "Harper Lee", "year": 1960, "genre": "Fiction", "rating": 4.9, "language":"English", "country":"USA"},
    ),
    Document(
        page_content="A crisp, engaging story with vibrant characters and a compelling plot",
        metadata={"title":"The Alchemist", "author": "Paulo Coelho", "year": 1988, "genre": "Adventure", "rating": 4.4, "language":"Portuguese", "country":"Brazil"},
    ),
    Document(
        page_content="A rich, complex narrative set in a dystopian future with strong thematic elements",
        metadata={"title":"1984", "author": "George Orwell", "year": 1949, "genre": "Dystopian", "rating": 4.7, "language":"English", "country":"UK"},
    ),
    Document(
        page_content="An intense, gripping story with dark themes and intricate plot twists",
        metadata={"title":"Gone Girl", "author": "Gillian Flynn", "year": 2012, "genre": "Thriller", "rating": 4.3, "language":"English", "country":"USA"},
    ),
    Document(
        page_content="An exotic, enchanting tale with rich descriptions and an intricate plot",
        metadata={"title":"One Hundred Years of Solitude", "author": "Gabriel García Márquez", "year": 1967, "genre": "Magical Realism", "rating": 4.8, "language":"Spanish", "country":"Colombia"},
    ),
    # ... (add more book documents as needed)
]

metadata_field_info = [
    AttributeInfo(
        name="title",
        description="The title of the book",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="author",
        description="The author of the book",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the book was published",
        type="integer",
    ),
    AttributeInfo(
        name="genre",
        description="The genre of the book",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="rating",
        description="The rating of the book (1-5 scale)",
        type="float",
    ),
    AttributeInfo(
        name="language",
        description="The language the book is written in",
        type="string",
    ),
    AttributeInfo(
        name="country",
        description="The country the author is from",
        type="string",
    ),
]



embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

document_content_description = "Brief description of the book"

llm = OpenAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

output = retriever.invoke("What are the titles of books written by authors from the USA?")
prompt = PromptTemplate.from_template(
    "Based on the following information, answer the question: {query}.\n\n{output}")

rag_chain = (
    {"output": retriever, "query": RunnablePassthrough()} |
    prompt |
    llm |
    StrOutputParser()
)

out = rag_chain.invoke("What are the titles of books written by authors from the Poland?")
print(out)