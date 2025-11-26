import spacy

print("\n=== PODOBIEŃSTWO SEMANTYCZNE ===")

# Załaduj większy model z wektorami słów
# python -m spacy download pl_core_news_md
try:
    nlp_md = spacy.load("pl_core_news_sm")

    doc1 = nlp_md("sztuczna inteligencja")
    doc2 = nlp_md("uczenie maszynowe")
    doc3 = nlp_md("samochód")

    print(f"Podobieństwo 'AI' vs 'ML': {doc1.similarity(doc2):.3f}")
    print(f"Podobieństwo 'AI' vs 'samochód': {doc1.similarity(doc3):.3f}")

    # Podobieństwo tokenów
    token1 = nlp_md("król")[0]
    token2 = nlp_md("królowa")[0]
    token3 = nlp_md("pies")[0]

    print(f"\nPodobieństwo 'król' vs 'królowa': {token1.similarity(token2):.3f}")
    print(f"Podobieństwo 'król' vs 'pies': {token1.similarity(token3):.3f}")

except:
    print("Zainstaluj większy model: python -m spacy download pl_core_news_md")