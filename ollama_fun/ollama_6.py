from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Inicjalizacja modelu
llm = Ollama(model="gemma3:12b")

# Prosty prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Napisz krótki artykuł o {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="machine learning")
print(result)