import ollama

# Prosty chat
response = ollama.chat(
    model='gemma3:12b',
    messages=[
        {
            'role': 'user',
            'content': 'Wyjaśnij czym jest rekurencja'
        }
    ]
)
print(response['message']['content'])