import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain.agents import create_agent
from langchain_core.tools import tool

from ingest import build_kb
if not os.path.exists("./stockfish_db"):
    st.info("First time setup: Indexing Stockfish documents...")
    try:
        build_kb()
        st.success("Indexing complete!")
    except Exception as e:
        st.error(f"Failed to build knowledge base: {e}")

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Stockfish Agent")
st.title("Stockfish Agent")

if "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in secrets.toml")
    st.stop()


# --- 2. EMBEDDINGS + DB (Load Once, Not Per Tool Call) ---
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
db = Chroma(persist_directory="./stockfish_db", embedding_function=embeddings)


# --- 3. TOOL ---
@tool
def search_stockfish_knowledge(query: str) -> str:
    """Search the local Stockfish database for technical or historical facts."""
    results = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])


tools = [search_stockfish_knowledge]


# --- 4. LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)


# --- 5. CREATE MODERN V1 AGENT ---
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a Stockfish expert. "
        "You MUST use the search_stockfish_knowledge tool before answering factual questions. "
        "If information is not found, say so."
    ),
)


# --- 6. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- 7. HANDLE INPUT ---
if user_query := st.chat_input("Ask about Stockfish..."):

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        response = agent.invoke({
            "messages": st.session_state.messages
        })

        output_text = response["messages"][-1].content

        st.markdown(output_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": output_text
        })