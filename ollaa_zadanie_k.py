import streamlit as st
import ollama

# Konfiguracja strony
st.set_page_config(
    page_title="Planer Podróży AI",
    page_icon="✈️",
    layout="centered"
)

# Tytuł aplikacji
st.title("✈️ Planer Podróży AI")
st.markdown("Stwórz plan swojej wymarzonej podróży z pomocą AI")

# Formularz wejściowy
with st.form("travel_form"):
    destination = st.text_input("Dokąd chcesz pojechać?", placeholder="np. Paryż, Tokio, Kraków")
    days = st.number_input("Ile dni?", min_value=1, max_value=30, value=3)
    interests = st.text_area(
        "Jakie masz zainteresowania?",
        placeholder="np. kultura, jedzenie, przyroda, sport",
        height=100
    )

    submit_button = st.form_submit_button("Wygeneruj plan podróży")


# Funkcja do generowania planu z Ollama SDK
def generate_travel_plan(destination, days, interests):
    prompt = f"""Stwórz szczegółowy plan podróży do {destination} na {days} dni.

Zainteresowania: {interests}

Proszę uwzględnij:
- Dzień po dniu atrakcje do odwiedzenia
- Rekomendacje restauracji
- Praktyczne wskazówki
- Szacunkowy budżet

Plan przedstaw w przejrzystej, punktowej formie."""

    try:
        # Prosty chat
        response = ollama.chat(
            model='gemma3:12b',
            messages=[
                {
                    'role': 'user',
                    'content': 'Wyjaśnij czym jest rekurencja'
                }
            ]
        )
        return response['message']['content']

    except Exception as e:
        return f"Błąd połączenia z Ollama: {str(e)}"


# Generowanie planu po kliknięciu przycisku
if submit_button:
    if not destination:
        st.error("Proszę podać miejsce docelowe!")
    else:
        with st.spinner("Tworzę plan podróży... To może chwilę potrwać..."):
            plan = generate_travel_plan(destination, days, interests)

            st.success("Plan podróży gotowy!")
            st.markdown("---")
            st.markdown("### Twój plan podróży:")
            st.markdown(plan)

            # Opcja pobrania planu
            st.download_button(
                label="📥 Pobierz plan jako TXT",
                data=plan,
                file_name=f"plan_podrozy_{destination.replace(' ', '_')}.txt",
                mime="text/plain"
            )