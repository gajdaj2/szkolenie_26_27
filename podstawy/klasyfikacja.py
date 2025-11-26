from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd

print("=== TEXTBLOB - ANALIZA SENTYMENTU ===\n")

# Przykładowe teksty
texts = [
    "I absolutely love this product! It's amazing and works perfectly!",
    "This is the worst purchase I've ever made. Terrible quality.",
    "It's okay, nothing special. Average product.",
    "Fantastic! Exceeded all my expectations. Highly recommend!",
    "Disappointing. Not worth the money at all.",
    "Pretty good, but could be better. Satisfied overall.",
]

# Analiza z domyślnym analizatorem (Pattern)
print("=== PATTERN ANALYZER (domyślny) ===\n")
results = []

for text in texts:
    blob = TextBlob(text)
    sentiment = blob.sentiment

    # Interpretacja
    if sentiment.polarity > 0.1:
        category = "POZYTYWNY"
    elif sentiment.polarity < -0.1:
        category = "NEGATYWNY"
    else:
        category = "NEUTRALNY"

    results.append({
        'text': text[:50] + '...' if len(text) > 50 else text,
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'category': category
    })

    print(f"Text: {text}")
    print(f"  Polarity: {sentiment.polarity:.3f} (od -1 do 1)")
    print(f"  Subjectivity: {sentiment.subjectivity:.3f} (od 0 do 1)")
    print(f"  Kategoria: {category}\n")

# DataFrame dla lepszej wizualizacji
df = pd.DataFrame(results)
print("\n=== PODSUMOWANIE ===")
print(df.to_string(index=False))