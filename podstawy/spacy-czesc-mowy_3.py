import spacy

# Załaduj model polski
nlp = spacy.load("pl_core_news_sm")

print("\n=== ROZPOZNAWANIE ENCJI - MODEL POLSKI ===")
tekst_ner = """
Adam Kowalski pracuje w firmie Google w Warszawie od 2020 roku.
Zarabia 15000 złotych miesięcznie. Jego email to adam@example.com.
Microsoft i Amazon są głównymi konkurentami Google.
Prezydent Andrzej Duda spotkał się z premierem Donaldem Tuskiem.
"""

doc_ner = nlp(tekst_ner)

# Sprawdź czy są jakieś encje
if doc_ner.ents:
    for ent in doc_ner.ents:
        print(f"{ent.text:20} | {ent.label_:15} | {spacy.explain(ent.label_)}")
else:
    print("Model polski ma ograniczone możliwości NER.")
    print("Wykryte encje:", len(doc_ner.ents))

# Grupowanie encji (jeśli są)
if doc_ner.ents:
    print("\n=== GRUPOWANIE ENCJI ===")
    from collections import defaultdict

    entities_by_type = defaultdict(list)
    for ent in doc_ner.ents:
        entities_by_type[ent.label_].append(ent.text)

    for label, entities in entities_by_type.items():
        print(f"\n{label}:")
        for ent in set(entities):
            print(f"  - {ent}")

