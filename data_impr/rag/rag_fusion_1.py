from operator import itemgetter

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from sympy.physics.units import temperature

load_dotenv()



loader = PyPDFLoader("../azure_ai_agents/files/OWU_i_Karta_produktu_Warta_Dla_Ciebie_i_Rodziny_od_14.04.2024.pdf")

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=60)

splits = text_splitter.split_documents(docs)


vectorstore  = Chroma.from_documents(splits, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """You are an AI language model assistant tasked with generating seach queries for a vector search engine.
The user has a question: "{question}"
Your goal/task is to create five variations of this {question} that capture different aspects of the user's intent. These variations will help the search engine retrieve relevant documents even if they don't use the exact keywords as the original question.
Provide these alternative questions, each on a new line.**
Original question: {question}"""

rag_fusion_prompt_template = ChatPromptTemplate.from_template(template)

generate_queries = (rag_fusion_prompt_template
                    |ChatOpenAI(temperature=0)
                    |StrOutputParser()
                    |StrOutputParser()
                    |(lambda x: x.split("\n"))
                    )


def reciprocal_rank_function(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a unique string identifier
            doc_str = str(doc)  # Simple string conversion
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (doc, score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


question = "Czy mam w ubezpieczeniu dostawę leków ?"
retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_function
docs = retrieval_chain.invoke({"question": question})
len(docs)

template = """Answer the following question based on this context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

out = final_rag_chain.invoke({"question":question})

print(out)