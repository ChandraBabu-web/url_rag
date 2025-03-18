import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter  

# Load environment variables from .env
load_dotenv()

# URL processing function
def process_input(urls, question):
    model_local = Ollama(model="mistral")
    
    # Convert string of URLs into a list and load documents
    urls_list = [url.strip() for url in urls.split("\n") if url.strip()]
    docs = []
    
    # Get FireCrawl API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        st.error("FIRECRAWL_API_KEY environment variable not set")
        return "Missing API Key."

    print(f"URLs to scrape: {urls_list}")
    print(f"Using FireCrawl API Key: {api_key}")

    for url in urls_list:
        try:
            print(f"Scraping {url}...")
            loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
            loaded_docs = loader.load()
            print(f"Loaded {len(loaded_docs)} documents from {url}")
            docs.extend(loaded_docs)
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")
            print(f"Error loading {url}: {e}")
    
    if not docs:
        print("No valid documents found.")
        return "No valid documents found."

    # Display document metadata
    print(f"Total documents loaded: {len(docs)}")
    for i, doc in enumerate(docs[:3]):  # Show a sample of 3 docs
        print(f"Doc {i+1} content (first 500 chars): {doc.page_content[:500]}")

    # Split the text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)
    print(f"Total document chunks after splitting: {len(doc_splits)}")

    # Convert text chunks into embeddings and store in the vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    
    
    
    retriever = vectorstore.as_retriever()

    # RAG processing
    after_rag_template = """Answer the question based only on the following information:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    print(f"Querying vector store with: {question}")
    response = after_rag_chain.invoke(question)
    print(f"Response: {response}")

    return response

# Streamlit UI
st.title("Website Query with Ollama (FireCrawl)")
st.write("Enter URLs (one per line) and a question to query the documents.")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# Process input on button click
if st.button('Query URLs'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)
