import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)

load_dotenv()

# ----------------------------
# Tools
# ----------------------------

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun()

tools = [search, arxiv, wiki]

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🔎 LangChain v1 Search Agent")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! How can I help you?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------
# Chat
# ----------------------------

if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
    )

    with st.chat_message("assistant"):

        placeholder = st.empty()

        final_response = ""

        for update in agent.stream(
            {
                "messages": st.session_state.messages
            },
            stream_mode="updates",
        ):

            st.write(update)

            if "model" in update:
                msgs = update["model"]["messages"]

                if msgs:
                    final_response = msgs[-1].content

        placeholder.write(final_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_response,
            }
        )