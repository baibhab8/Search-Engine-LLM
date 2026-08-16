import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# ----------------------------
# Tool
# ----------------------------

search = DuckDuckGoSearchRun()

tools = [search]


# ----------------------------
# System Prompt
# ----------------------------

SYSTEM_PROMPT = """
You are a reliable web research assistant.

For current, recent, ranking, financial, statistical, historical,
or time-sensitive questions, ALWAYS use the web search tool.

Do not answer factual questions from memory when web search can
verify the information.

For rankings and numerical information:
- Search the web first.
- Prefer authoritative sources.
- Prefer one reliable source containing the complete ranking.
- Do not combine information from different dates or sources.
- Do not mix annual rankings with real-time rankings.

Pay close attention to dates mentioned by the user.

If search results conflict, perform another search and compare
the information.

Never invent facts, numbers, rankings, dates, or sources.

Give a concise and accurate answer.
Mention the important source or date when relevant.
"""


# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🔎 LangChain v1 Search Agent")

st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)


# ----------------------------
# Chat History
# ----------------------------

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

    if not api_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    with st.chat_message("assistant"):

        final_response = ""

        for update in agent.stream(
            {
                "messages": st.session_state.messages
            },
            stream_mode="updates"
        ):

            if "model" in update:

                messages = update["model"]["messages"]

                if messages:
                    final_response = messages[-1].content

        st.write(final_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_response
        }
    )
