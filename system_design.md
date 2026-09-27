PDFUpload--> pypdfloader --> Page aware documents are created (documents are divided intp pages)--> Recursive character text splitter --> Sentence transformer (Creating embedding )--> store in ChromaDB --> Similarity search --> Gemini model for LM call --> Academic answer with the proper citation sources




Ingestion-
    Chunking from pdf
    Embedding 
    indexing
    storing embedding in Chromadb

Supporting:
Configuration
    gemini flash LLM
    vectordb- Chromadb



Front end:
    Steramlit
    Q&A functionality
    Revision
        Topic summary
        Falshcard
