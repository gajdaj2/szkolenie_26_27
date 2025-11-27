import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
import tempfile
import os
from pathlib import Path

# Konfiguracja strony
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Chatbot - Zapytaj o swoje dokumenty")
st.markdown("---")

# Inicjalizacja session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# Funkcja do przetwarzania dokumentów
def process_documents(uploaded_files, api_key, chunk_size, chunk_overlap):
    """Przetwarza załadowane pliki PDF i tworzy bazę wektorową"""
    with st.spinner("📖 Ładuję i przetwarszam dokumenty..."):
        try:
            all_docs = []

            # Przetwarzanie każdego pliku
            for uploaded_file in uploaded_files:
                # Zapisz plik tymczasowo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Załaduj PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                st.info(f"📄 {uploaded_file.name}: załadowano {len(pages)} stron")

                # Konwertuj na dokumenty
                for page in pages:
                    all_docs.append(
                        Document(
                            page_content=page.page_content,
                            metadata={
                                **page.metadata,
                                "filename": uploaded_file.name
                            }
                        )
                    )

                # Usuń tymczasowy plik
                os.unlink(tmp_path)

            # Podziel dokumenty na chunki
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            documents_split = splitter.split_documents(all_docs)

            st.info(f"✂️ Podzielono na {len(documents_split)} fragmentów")

            # Utwórz embeddingi i bazę wektorową
            embeddings = OpenAIEmbeddings(api_key=api_key)

            vectordb = Chroma.from_documents(
                documents=documents_split,
                embedding=embeddings,
                collection_name="rag_chatbot"
            )

            # Zapisz w session state
            st.session_state.vectordb = vectordb
            st.session_state.documents_loaded = True
            st.session_state.api_key = api_key

            st.success(f"✅ Dokumenty przetworzone! Zaindeksowano {len(all_docs)} stron z {len(uploaded_files)} plików.")

        except Exception as e:
            st.error(f"❌ Błąd podczas przetwarzania: {str(e)}")


# Funkcja do generowania odpowiedzi RAG
def generate_rag_response(question, api_key, model_name, temperature, k_results):
    """Generuje odpowiedź używając RAG"""
    try:
        # Inicjalizacja modelu
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=temperature,
            model=model_name
        )

        # Template promptu
        template = """
Odpowiedz na pytanie na podstawie dostarczonego kontekstu z dokumentów.

Kontekst:
{kontekst}

Pytanie: {pytanie}

Instrukcje:
- Odpowiadaj wyłącznie na podstawie dostarczonego kontekstu
- Jeśli nie znajdziesz odpowiedzi w kontekście, powiedz to wprost
- Cytuj konkretne fragmenty z dokumentów jeśli to możliwe
- Bądź precyzyjny i zwięzły

Odpowiedź:
"""

        prompt = PromptTemplate.from_template(template=template)

        # Retriever
        retrieval = st.session_state.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_results}
        )

        # Łańcuch RAG
        chain = (
                {
                    "kontekst": retrieval,
                    "pytanie": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
        )

        # Generuj odpowiedź
        response = chain.invoke(question)

        # Pobierz dokumenty źródłowe
        source_docs = retrieval.invoke(question)

        return response, source_docs

    except Exception as e:
        st.error(f"Błąd podczas generowania odpowiedzi: {str(e)}")
        return None, None


# Sidebar z konfiguracją
with st.sidebar:
    st.header("⚙️ Konfiguracja")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Wprowadź swój klucz API OpenAI"
    )

    st.subheader("📄 Załaduj dokumenty PDF")
    uploaded_files = st.file_uploader(
        "Wybierz pliki PDF",
        type=['pdf'],
        accept_multiple_files=True,
        help="Możesz załadować wiele plików PDF"
    )

    st.subheader("🔧 Parametry RAG")

    chunk_size = st.slider(
        "Rozmiar chunka",
        min_value=256,
        max_value=2048,
        value=1024,
        step=128,
        help="Rozmiar pojedynczego fragmentu tekstu"
    )

    chunk_overlap = st.slider(
        "Nakładanie chunków",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Ile znaków ma się nakładać między chunkami"
    )

    k_results = st.slider(
        "Liczba wyników wyszukiwania",
        min_value=1,
        max_value=10,
        value=3,
        help="Ile najlepszych fragmentów pobrać z bazy"
    )

    st.subheader("🤖 Parametry modelu")

    model_name = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Wyższa wartość = bardziej kreatywne odpowiedzi"
    )

    st.markdown("---")

    if st.button("🔄 Przetwórz dokumenty", type="primary"):
        if not api_key:
            st.error("⚠️ Wprowadź klucz API OpenAI!")
        elif not uploaded_files:
            st.error("⚠️ Załaduj przynajmniej jeden plik PDF!")
        else:
            process_documents(uploaded_files, api_key, chunk_size, chunk_overlap)

    if st.button("🗑️ Wyczyść historię czatu"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Resetuj bazę dokumentów"):
        st.session_state.vectordb = None
        st.session_state.documents_loaded = False
        st.session_state.messages = []
        st.success("✅ Baza dokumentów została zresetowana")
        st.rerun()

# Layout główny - dwie kolumny
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Czat")

    # Wyświetlanie historii konwersacji
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Wyświetl źródła jeśli są dostępne
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Źródła"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Źródło {i}:**")
                        st.markdown(f"*Plik: {source.metadata.get('filename', 'N/A')}*")
                        st.markdown(f"*Strona: {source.metadata.get('page', 'N/A')}*")
                        st.text(source.page_content[:300] + "...")
                        st.markdown("---")

    # Input użytkownika
    if not st.session_state.documents_loaded:
        st.info("👆 Załaduj dokumenty PDF w sidebarze, aby rozpocząć czat")
    else:
        if prompt := st.chat_input("Zadaj pytanie o dokumenty..."):
            # Sprawdź API key
            if not api_key:
                st.error("⚠️ Wprowadź klucz API OpenAI w sidebarze!")
                st.stop()

            # Dodaj wiadomość użytkownika
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generuj odpowiedź
            with st.chat_message("assistant"):
                with st.spinner("🔍 Szukam w dokumentach..."):
                    response, sources = generate_rag_response(
                        prompt,
                        api_key,
                        model_name,
                        temperature,
                        k_results
                    )

                    if response:
                        st.markdown(response)

                        # Dodaj odpowiedź do historii
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })

                        # Wyświetl źródła
                        if sources:
                            with st.expander("📚 Źródła"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Źródło {i}:**")
                                    st.markdown(f"*Plik: {source.metadata.get('filename', 'N/A')}*")
                                    st.markdown(f"*Strona: {source.metadata.get('page', 'N/A')}*")
                                    st.text(source.page_content[:300] + "...")
                                    st.markdown("---")

with col2:
    st.subheader("📊 Status")

    if st.session_state.documents_loaded:
        st.success("✅ Dokumenty załadowane")

        # Statystyki
        if st.session_state.vectordb:
            st.metric("Fragmentów w bazie",
                      st.session_state.vectordb._collection.count())
    else:
        st.warning("⏳ Brak załadowanych dokumentów")

    st.markdown("---")
    st.subheader("ℹ️ Jak używać?")
    st.markdown("""
    1. Wprowadź klucz API OpenAI
    2. Załaduj pliki PDF
    3. Kliknij "Przetwórz dokumenty"
    4. Zadawaj pytania o zawartość dokumentów
    """)

    st.markdown("---")
    st.subheader("🎯 Przykładowe pytania")
    st.markdown("""
    - Jakie są główne tematy dokumentu?
    - Czy w dokumencie jest informacja o...?
    - Podsumuj sekcję dotyczącą...
    - Jakie są kluczowe daty/liczby?
    """)

# Stopka
st.markdown("---")
st.caption("💡 RAG (Retrieval Augmented Generation) pozwala chatbotowi odpowiadać na podstawie Twoich dokumentów")
