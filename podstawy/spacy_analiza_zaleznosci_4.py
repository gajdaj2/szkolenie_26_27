import spacy
from spacy import displacy

print("\n=== ANALIZA ZALEŻNOŚCI ===")

nlp = spacy.load("pl_core_news_sm")

zdanie = nlp("Sztuczna inteligencja rewolucjonizuje przemysł technologiczny.")

for token in zdanie:
    print(f"{token.text:15} <-- {token.dep_:10} -- {token.head.text}")

# Wizualizacja drzewa zależności
html = displacy.render(zdanie, style="dep", page=True, options={'distance': 120})
with open("dependencies.html", "w", encoding="utf-8") as f:
    f.write(html)

# Znajdowanie podmiotów i dopełnień
print("\n=== PODMIOTY I ORZECZENIA ===")
for token in zdanie:
    if token.dep_ == "nsubj":
        print(f"Podmiot: {token.text}")
    if token.dep_ == "ROOT":
        print(f"Orzeczenie: {token.text}")
    if token.dep_ == "obj":
        print(f"Dopełnienie: {token.text}")