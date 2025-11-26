import ollama

response = ollama.chat(
    model='gemma3:12b',
    messages=[{'role': 'user', 'content': 'Opowiedz żart'}],
    options={
        'temperature': 0.8,      # Kreatywność (0.0-2.0)
        'top_p': 0.9,           # Nucleus sampling
        'top_k': 40,            # Top-k sampling
        'num_predict': 100,     # Max długość odpowiedzi
    }
)

print(response['message']['content'])