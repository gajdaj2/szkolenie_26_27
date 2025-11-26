import spacy

# Załadowanie modelu
nlp = spacy.load("pl_core_news_sm")

# Przetworzenie tekstu
tekst = """
Sztuczna inteligencja zmienia świat technologii. 
Google i Microsoft inwestują miliardy dolarów w AI.
Warszawa, 15 listopada 2024 roku.
"""

doc = nlp(tekst)

# Tokenizacja
print("=== TOKENY ===")
for token in doc:
    print(f"{token.text:15} | POS: {token.pos_:10} | Lemma: {token.lemma_}")

# Zdania
print("\n=== ZDANIA ===")
for i, sent in enumerate(doc.sents, 1):
    print(f"{i}. {sent.text.strip()}")