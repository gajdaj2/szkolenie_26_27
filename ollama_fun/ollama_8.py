import ollama

# Lista modeli
models = ollama.list()
for model in models['models']:
    print(model['name'])

# Pobieranie modelu
ollama.pull('gemma2:12b')

# Usuwanie modelu
ollama.delete('gemma2:12b')

# Informacje o modelu
info = ollama.show('gemma2:12b')
print(info)