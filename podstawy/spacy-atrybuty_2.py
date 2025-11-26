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

print("\n=== SZCZEGÓŁOWA ANALIZA TOKENÓW ===")
for token in doc[:15]:
    print(f"""
    Text: {token.text}
    Lemma: {token.lemma_}
    POS: {token.pos_}
    Tag: {token.tag_}
    Dep: {token.dep_}
    Shape: {token.shape_}
    is_alpha: {token.is_alpha}
    is_stop: {token.is_stop}
    """)

# Filtrowanie tokenów
print("\n=== TYLKO RZECZOWNIKI ===")
rzeczowniki = [token.text for token in doc if token.pos_ == "NOUN"]
print(rzeczowniki)

print("\n=== TYLKO CZASOWNIKI ===")
czasowniki = [token.lemma_ for token in doc if token.pos_ == "VERB"]
print(czasowniki)

print("\n=== BEZ STOP WORDS ===")
bez_stopwords = [token.text for token in doc if not token.is_stop and token.is_alpha]
print(bez_stopwords)