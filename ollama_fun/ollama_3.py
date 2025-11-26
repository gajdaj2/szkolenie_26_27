import ollama

messages = [
    {'role': 'user', 'content': 'Jak masz na imię?'},
]

# Pierwsza odpowiedź
response = ollama.chat(model='gemma3:12b', messages=messages)
messages.append({'role': 'assistant', 'content': response['message']['content']})

# Kontynuacja rozmowy
messages.append({'role': 'user', 'content': 'A jakie są twoje mocne strony?'})
response = ollama.chat(model='gemma3:12b', messages=messages)
print(response['message']['content'])