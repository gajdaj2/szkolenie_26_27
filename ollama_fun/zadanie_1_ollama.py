import ollama


def utworz_przepis(danie: str) -> str:
    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'system',
                'content': 'Jesteś doświadczonym szefem kuchni. Twórz szczegółowe przepisy kulinarne w języku polskim.'
            },
            {
                'role': 'user',
                'content': f'Stwórz przepis na: {danie}'
            }
        ]
    )
    return response['message']['content']


def analizuj_gramature(przepis: str) -> str:
    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'system',
                'content': 'Jesteś dietetykiem. Analizujesz przepisy i podajesz szczegółowe informacje o gramaturze składników.'
            },
            {
                'role': 'user',
                'content': f'Na podstawie tego przepisu:\n\n{przepis}\n\nPodaj dokładną gramaturę wszystkich składników w gramach.'
            }
        ]
    )
    return response['message']['content']


def oszacuj_czas(przepis: str) -> str:
    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'system',
                'content': 'Jesteś ekspertem w zarządzaniu czasem w kuchni. Precyzyjnie szacujesz czas potrzebny na przygotowanie potraw.'
            },
            {
                'role': 'user',
                'content': f'Na podstawie tego przepisu:\n\n{przepis}\n\nOkreśl dokładny czas potrzebny na:\n1. Przygotowanie składników\n2. Gotowanie/pieczenie\n3. Łączny czas'
            }
        ]
    )
    return response['message']['content']


def chain_przepis_kompletny(danie: str) -> dict:
    """
    Główna funkcja łańcucha - tworzy przepis i analizuje go.
    """
    print("=" * 60)
    print(f"Tworzenie przepisu na: {danie}")
    print("=" * 60)

    # Krok 1: Tworzenie przepisu
    print("\n🍳 KROK 1: Generowanie przepisu...")
    przepis = utworz_przepis(danie)
    print("\n" + przepis)

    # Krok 2: Analiza gramatury
    print("\n" + "=" * 60)
    print("⚖️  KROK 2: Analiza gramatury składników...")
    print("=" * 60)
    gramatura = analizuj_gramature(przepis)
    print("\n" + gramatura)

    # Krok 3: Oszacowanie czasu
    print("\n" + "=" * 60)
    print("⏱️  KROK 3: Oszacowanie czasu przygotowania...")
    print("=" * 60)
    czas = oszacuj_czas(przepis)
    print("\n" + czas)

    return {
        "przepis": przepis,
        "gramatura": gramatura,
        "czas": czas
    }


def zapisz_do_pliku(wynik: dict, nazwa_pliku: str = "przepis_kompletny.txt"):
    """Zapisuje wyniki do pliku."""
    with open(nazwa_pliku, "w", encoding="utf-8") as f:
        f.write("PRZEPIS\n")
        f.write("=" * 60 + "\n")
        f.write(wynik["przepis"] + "\n\n")
        f.write("GRAMATURA\n")
        f.write("=" * 60 + "\n")
        f.write(wynik["gramatura"] + "\n\n")
        f.write("CZAS PRZYGOTOWANIA\n")
        f.write("=" * 60 + "\n")
        f.write(wynik["czas"] + "\n")
    print(f"\n✅ Kompletny przepis zapisany do pliku '{nazwa_pliku}'")


# Przykład użycia
if __name__ == "__main__":
    # Wybierz danie
    danie = "carbonara"

    # Uruchom chain
    wynik = chain_przepis_kompletny(danie)
