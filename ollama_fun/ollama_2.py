import ollama

stream = ollama.chat(
    model='gemma3:12b',
    messages=[{'role': 'user', 'content': 'Napisz krótką historię'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)