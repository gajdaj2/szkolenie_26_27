from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()


personal_finance_template = """You are a personal finance expert with extensive knowledge of budgeting, investing, and financial planning. You offer clear and practical advice on managing money and making sound financial decisions.

Here is a question:
{query}"""

book_review_template = """You are an experienced book critic with extensive knowledge of literature, genres, and authors. You provide thoughtful and analytical reviews and insights about books.

Here is a question:
{query}"""

health_fitness_template = """You are a certified health and fitness expert with a deep understanding of nutrition, exercise routines, and wellness strategies. You offer practical and evidence-based advice about health and fitness.

Here is a question:
{query}"""

travel_guide_template = """You are a seasoned travel expert with extensive knowledge of destinations, travel tips, and cultural insights. You provide detailed and useful advice about travel.

Here is a question:
{query}"""


classification_template = PromptTemplate.from_template(
    """You are good at classifying a question.
    Given the user question below, classify it as either being about personal finance, book reviews, health & fitness, or travel guides.

    <If the question is about budgeting, investing, or financial planning, classify it as 'Personal Finance'>
    <If the question is about literature, genres, or authors, classify it as 'Book Review'>
    <If the question is about nutrition, exercise routines, or wellness strategies, classify it as 'Health & Fitness'>
    <If the question is about destinations, travel tips, or cultural insights, classify it as 'Travel Guide'>

    <question>
    {question}
    </question>

    Classification:"""
)


classification_chain = classification_template | ChatOpenAI() | StrOutputParser()


def prompt_router(input_query):
    classification = classification_chain.invoke({"question": input_query["query"]})

    if classification == "Personal Finance":
        print("Using PERSONAL FINANCE")
        return PromptTemplate.from_template(personal_finance_template)
    elif classification == "Book Review":
        print("Using BOOK REVIEW")
        return PromptTemplate.from_template(book_review_template)
    elif classification == "Health & Fitness":
        print("Using HEALTH & FITNESS")
        return PromptTemplate.from_template(health_fitness_template)
    elif classification == "Travel Guide":
        print("Using TRAVEL GUIDE")
        return PromptTemplate.from_template(travel_guide_template)
    else:
        print("Unexpected classification:", classification)
        return None


input_query = {"query": "What are effective strategies for losing weight?"}
prompt = prompt_router(input_query)

input_query = {"query": "What are the must-see attractions in USA?"}

if prompt:
    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | ChatOpenAI()
        | StrOutputParser()
    )

    response = chain.invoke(input_query["query"])

    print(response)
else:
    print("Could not determine appropriate LLM for the query.")