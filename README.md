# Agentic AI Demo

A Streamlit-based application demonstrating agentic AI capabilities using LangGraph, LangChain, and multiple AI agents for research tasks.

## Features

- Research Agent: Uses Tavily API to gather information
- Drafting Agent: Processes research results using Google's Gemini model
- Interactive Streamlit interface

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your API keys in environment variables:
   - GOOGLE_API_KEY
   - TAVILY_API_KEY

3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Enter your research query in the text input
2. Click 'Start Research'
3. Wait for the agents to gather and process information
4. View the final synthesized answer
