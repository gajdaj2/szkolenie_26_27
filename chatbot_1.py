import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Konfiguracja strony
st.set_page_config(
    page_title="Prosty Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Prosty Chatbot z LangChain")
st.markdown("---")

# Inicjalizacja session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )

# Sidebar z konfiguracją
with st.sidebar:
    st.header("⚙️ Konfiguracja")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Wprowadź swój klucz API OpenAI"
    )

    model_name = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Wyższa wartość = bardziej kreatywne odpowiedzi"
    )

    system_prompt = st.text_area(
        "Prompt systemowy",
        value="Jesteś pomocnym asystentem AI. Odpowiadaj zwięźle i konkretnie.",
        height=100
    )

    if st.button("🗑️ Wyczyść historię"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()


# Funkcja do inicjalizacji modelu
def get_llm(api_key, model_name, temperature):
    if not api_key:
        st.error("⚠️ Wprowadź klucz API OpenAI w sidebarze!")
        st.stop()

    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature
    )


# Funkcja do tworzenia łańcucha konwersacji
def create_conversation_chain(llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=prompt,
        verbose=False
    )

    return chain


# Wyświetlanie historii konwersacji
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input użytkownika
if prompt := st.chat_input("Napisz wiadomość..."):
    # Dodaj wiadomość użytkownika do historii
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Wyświetl wiadomość użytkownika
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generuj odpowiedź
    with st.chat_message("assistant"):
        with st.spinner("Myślę..."):
            try:
                # Inicjalizacja modelu i łańcucha
                llm = get_llm(api_key, model_name, temperature)
                chain = create_conversation_chain(llm, system_prompt)

                # Generuj odpowiedź
                response = chain.predict(input=prompt)

                # Wyświetl odpowiedź
                st.markdown(response)

                # Dodaj odpowiedź do historii
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(f"Wystąpił błąd: {str(e)}")

# Informacje w stopce
st.markdown("---")
st.caption("💡 Tip: Możesz dostosować zachowanie chatbota w sidebarze")