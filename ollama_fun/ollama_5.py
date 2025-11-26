import ollama

with open('imaga.jpg', 'rb') as file:
    response = ollama.chat(
        model='gemma3:12b',  # Gemma2 nie wspiera vision, użyj llava
        messages=[{
            'role': 'user',
            'content': 'Jaki kolor jest na obrazku ?',
            'images': [file.read()]
        }]
    )
print(response['message']['content'])