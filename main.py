import os
import streamlit as st
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# ✅ Set API Keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyBcwt8ImzAjGJejImwatp3RWAYmhGxwshQ"
os.environ["TAVILY_API_KEY"] = "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"

# Initialize Tavily Client
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Initialize PaLM (or other free Google models)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)  # Example using text-bison model for free usage


# -----------------------
# Define State Schema for LangGraph
# -----------------------
class ResearchState(TypedDict):
    query: str
    research_results: Optional[Any]
    final_answer: Optional[str]


# Initialize LangGraph with the defined schema
graph = StateGraph(ResearchState)


# -----------------------
# Agent 1: Research Agent
# -----------------------
def research_agent(state):
    query = state["query"]
    st.write(f"[🔍 Research Agent] Querying: {query}")

    # Using Tavily SDK directly (per docs)
    results = tavily_client.search(
        query,
        search_depth="advanced",
        max_results=7,
        include_answer=True
    )

    state["research_results"] = results
    return state


# -----------------------
# Agent 2: Drafting Agent
# -----------------------
def draft_agent(state):
    query = state["query"]
    research_results = state["research_results"]

    # Extract only the content from the results
    formatted = ""
    for i, result in enumerate(research_results["results"], 1):
        # Extract only the 'content' field from each result
        formatted += f"{i}. {result['content']}\n\n"

    prompt = PromptTemplate.from_template("""
You are a helpful AI researcher.

Using the following research results, write a clear and informative response to the query:

"{query}"

Research Results:
{formatted}

Answer:
""")

    chain = prompt | llm
    answer = chain.invoke({"query": query, "formatted": formatted})

    state["final_answer"] = answer
    return state


# -----------------------
# LangGraph Flow Setup
# -----------------------
graph.add_node("Research", research_agent)
graph.add_node("Draft", draft_agent)
graph.set_entry_point("Research")
graph.add_edge("Research", "Draft")
graph.add_edge("Draft", END)

# Compile the graph
app = graph.compile()


# -----------------------
# Streamlit Interface
# -----------------------
def main():
    st.title("Deep Research AI Agentic System")

    st.write(
        "This system allows you to ask research-related questions, and the AI will gather information and draft an answer.")

    query = st.text_input("🔎 Enter your research topic:")

    if st.button('Start Research'):
        if query:
            with st.spinner("Researching..."):
                result = app.invoke({"query": query})

            st.success("Research Complete!")
            st.subheader("🧠 Final Answer:")
            st.write(result["final_answer"].content)
        else:
            st.error("Please enter a research topic.")


if __name__ == "__main__":
    main()