import spacy

print("\n=== LEMMATYZACJA ===")

nlp = spacy.load("pl_core_news_sm")
tekst_lemma = """
Kupiłem nowe książki. Czytam je codziennie.
Dzieci biegały po parku. Najlepsi programiści pracują w Google.
"""

doc_lemma = nlp(tekst_lemma)

for token in doc_lemma:
    if token.is_alpha:
        print(f"{token.text:15} -> {token.lemma_}")