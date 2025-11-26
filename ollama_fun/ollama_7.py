from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:12b")

response = llm.generate(["Napisz krótki artykuł o sztucznej inteligencji."])
print(response.generations[0][0].text)


prompt = "Napisz krótki artykuł o zaletach programowania w {language}."

template = PromptTemplate.from_template(template=prompt)

formatted_prompt = template.format(language="Python")

chain = template|llm|StrOutputParser()

result = chain.invoke({"language": "Python"})
print(result)
