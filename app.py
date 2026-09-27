import os
from pathlib import Path
import streamlit as st
from src.config import (app_title, chroma_dir, get_secret)
from src.pdf_processor import load_and_split_pdfs
from src.vector_store import create_vector_store, load_vector_store
from src.rag_engine import RAGEngine
from src.flashcards import generate_flashcards, generate_summary

st.set_page_config(page_title = app_title, page_icon = "🎶", layout = "wide", )
st.title("PDF Processor")
st.caption("you can ask any question about your pdf")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("Configuration")
    gemini_key = get_secret("GEMINI_API_KEY")
    if gemini_key:
        st.success("Gemini API key detected")
    else:
        st.warning("Gemini API key is not configured, pls add in you toml file locally or streamlit cloud secrets")

    st.divider()

    if st.button("clear current session", use_container_width = True):
        st.session_state.vector_store = None
        st.session_state.rag_engine = None
        st.session_state.chat_history = []
        st.session_state.indexed_files = []

tab_upload, tab_chat, tab_revision = st.tabs(["Upload Index","Academic Q&A", "Revsion"])
with tab_upload:
    st.subheader("Upload your material here")
    uploaded_files = st.file_uploader("Upload one or more pdf files", type = ["pdf"], accept_multiple_files = True, help = "pdfs are parsed page by page and indexed for semantic search",)
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("chunk size", min_value = 300, max_value = 3000, value = 1000, step = 100,)
    with col2:
            chunk_overlap = st.number_input("chunk overlap", min_value = 0, max_value = 800, value = 200, step = 50,)
    if st.button("build knowledge base", type = "primary", use_container_width = True, disabled = not uploaded_files,):
        if not gemini_key:
            st.error("configure your gemini api key before indexing")
            st.stop

        try:
            with st.spinner("Reading PDFs and creating chunks..."):
                documents = load_and_split_pdfs(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            if not documents:
                st.error("No readable text was found in the uploaded PDFs.")
                st.stop()

            with st.spinner("Creating embeddings and ChromaDB index..."):
                vector_store = create_vector_store(
                    documents=documents,
                    persist_directory=chroma_dir,
                )

            st.session_state.vector_store = vector_store
            st.session_state.rag_engine = RAGEngine(
                vector_store=vector_store,
                api_key=gemini_key,
            )
            st.session_state.chat_history = []
            st.session_state.indexed_files = [
                file.name for file in uploaded_files
            ]

            st.success(
                f"Indexed {len(uploaded_files)} PDF(s) into "
                f"{len(documents)} searchable chunks."
            )

        except Exception as exc:
            st.exception(exc)

    st.divider()

    if st.session_state.indexed_files:
        st.subheader("Indexed documents")
        for filename in st.session_state.indexed_files:
            st.write(f" {filename}")

    elif Path(chroma_dir).exists():
        st.info(
            "A previous ChromaDB directory exists. "
            "Use the reload button below if you want to reuse it."
        )

        if st.button(" Load existing index"):
            try:
                vector_store = load_vector_store(chroma_dir)
                st.session_state.vector_store = vector_store

                if gemini_key:
                    st.session_state.rag_engine = RAGEngine(
                        vector_store=vector_store,
                        api_key=gemini_key,
                    )

                st.success("Existing ChromaDB index loaded.")
            except Exception as exc:
                st.exception(exc)

        
with tab_chat:
    st.subheader("Ask your academic assistant")

    if st.session_state.rag_engine is None:
        st.info("Upload and index your PDFs first.")
    else:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander(" Sources"):
                        for source in message["sources"]:
                            st.write(
                                f" {source['filename']} — "
                                f"Page {source['page']}"
                            )

        question = st.chat_input(
            "Ask a question about your course material..."
        )

        if question:
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching your documents..."):
                    try:
                        result = st.session_state.rag_engine.ask(question)

                        st.markdown(result["answer"])

                        if result["sources"]:
                            with st.expander("Sources"):
                                for source in result["sources"]:
                                    st.write(
                                        f" {source['filename']} — "
                                        f"Page {source['page']}"
                                    )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            }
                        )

                    except Exception as exc:
                        st.error(f"Unable to answer the question: {exc}")

with tab_revision:
    st.subheader("Revision tools")

    if st.session_state.rag_engine is None:
        st.info("Upload and index your PDFs first.")
    else:
        revision_mode = st.radio(
            "Choose a revision tool",
            ["Chapter / topic summary", "Flashcards"],
            horizontal=True,
        )

        topic = st.text_input(
            "Topic",
            placeholder="e.g. Photosynthesis, Database Normalization, World War II",
        )

        if st.button(
            "✨ Generate",
            type="primary",
            disabled=not topic.strip(),
        ):
            try:
                if revision_mode == "Chapter / topic summary":
                    with st.spinner("Retrieving relevant material and summarizing..."):
                        result = generate_summary(
                            st.session_state.rag_engine,
                            topic,
                        )

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("Sources"):
                            for source in result["sources"]:
                                st.write(
                                    f"{source['filename']} — "
                                    f"Page {source['page']}"
                                )

                else:
                    with st.spinner("Creating flashcards..."):
                        result = generate_flashcards(
                            st.session_state.rag_engine,
                            topic,
                        )

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("Sources"):
                            for source in result["sources"]:
                                st.write(
                                    f"{source['filename']} — "
                                    f"Page {source['page']}"
                                )
            except Exception as exc:
                st.error(f"Revision generation failed: {exc}")    