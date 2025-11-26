from textblob import TextBlob

print("\n=== TEXTBLOB ===")

text = """
TextBlob is a simple library for processing textual data.
It provides a simple API for common NLP tasks.
The sentiment analysis is quite accurate!
"""

blob = TextBlob(text)

# Zdania
print("=== ZDANIA ===")
for i, sentence in enumerate(blob.sentences, 1):
    print(f"{i}. {sentence}")

# Słowa
print(f"\n=== SŁOWA ===")
print(f"Wszystkie słowa: {blob.words[:10]}")

# POS tagging
print("\n=== POS TAGS ===")
for word, tag in blob.tags[:10]:
    print(f"{word:15} -> {tag}")

# Rzeczowniki
print(f"\n=== RZECZOWNIKI ===")
print(blob.noun_phrases)

# Analiza sentymentu
print("\n=== SENTIMENT ANALYSIS ===")
for sentence in blob.sentences:
    print(f"Text: {sentence}")
    print(f"  Polarity: {sentence.sentiment.polarity:.2f}")
    print(f"  Subjectivity: {sentence.sentiment.subjectivity:.2f}")

# Tłumaczenie (wymaga translate)
print("\n=== TRANSLATION ===")
try:
    polish_text = blob.translate(to='pl')
    print(f"PL: {polish_text}")
except:
    print("Tłumaczenie wymaga połączenia z internetem")

# Korekta pisowni
print("\n=== SPELLING CORRECTION ===")
misspelled = TextBlob("I havv goood speling")
print(f"Przed: {misspelled}")
print(f"Po: {misspelled.correct()}")

# N-gramy
print("\n=== N-GRAMY ===")
print(f"Bigramy: {blob.ngrams(n=2)[:5]}")