# Website Query with Ollama (RAG)

## Overview
This project is a Streamlit-based web application that allows users to query websites for information using the FireCrawl API. The application scrapes the provided URLs, processes the extracted text into vector embeddings using Hugging Face BGE embeddings, and performs a Retrieval-Augmented Generation (RAG) process to generate an AI-powered answer to the user’s query using Ollama with the Mistral model.

## Features
- **Scrape Websites:** Uses FireCrawl API to fetch content from user-provided URLs.
- **Text Processing:** Cleans and splits the content into chunks for efficient processing.
- **Vector Database:** Uses FAISS to store and retrieve document embeddings.
- **RAG Model:** Utilizes Ollama with the Mistral model to answer user queries based on retrieved data.
- **Streamlit UI:** Provides an interactive web-based interface for users to input URLs and questions.

## Installation
### Prerequisites
- Python 3.8+
- A FireCrawl API key (stored in `.env` file)
- Set up Ollama from the website: [Ollama Mistral](https://ollama.com/library/mistral)
- Run the following commands to pull the model:
  ```bash
  ollama pull mistral
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ChandraBabu-web/url_rag.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your FireCrawl API key:
   ```
   FIRECRAWL_API_KEY=your_api_key_here
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter URLs (one per line) in the text area.
2. Type a question related to the content of the URLs.
3. Click the **Query URLs** button.
4. The processed answer will be displayed in the output section.

## Project Structure
```
📂 project-directory
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── README.md            # Documentation
```

## Dependencies
- `streamlit`
- `python-dotenv`
- `langchain`
- `faiss`
- `Hugging Face BGE embeddings`
- `Ollama`

## Troubleshooting
- If the app does not start, ensure all dependencies are installed.
- If FireCrawl API requests fail, check the API key in the `.env` file.
- If the query returns an empty response, verify that the URLs contain valid content.

## Acknowledgments
- **LangChain** for text processing and retrieval
- **FireCrawl API** for web scraping
- **FAISS** for vector database management
- **Ollama** for LLM-powered query responses
- **Streamlit** for the user interface

