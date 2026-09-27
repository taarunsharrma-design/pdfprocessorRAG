from pathlib import Path # pick OS Used
from uuid import uuid4 # To generate random universally unique identifiers
from langchain_chroma import Chroma 
from src.embeddings import get_embeddings

COLLECTION_NAME = "academic_documents"

def create_vector_store(documents, persist_directory:str, )->Chroma:
    persist_path = Path(persist_directory) # Converting input string path into pathlib path object
    persist_path.mkdir(parents=True, exist_ok=True) # Creating the directory structure on disk
    embeddings = get_embeddings()
    collection_name = f"{COLLECTION_NAME}_{uuid4().hex[:8]}" #Creating randomized collection name
    vector_store = Chroma.from_documents(documents = documents, embedding = embeddings, collection_name= collection_name, persist_directory=str(persist_path), ) #Using langchain factory method to convert text Document Into a numerical vector And index them. 
    return vector_store

def load_vector_store(
        persist_directory:str, collection_name:str|None = None, 
)->Chroma:
    embeddings = get_embeddings()
    if collection_name is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(Path(persist_directory)))
        collections = client.list_collections()
        if not collections:
            raise ValueError("No chromadb collections found")
        collection_name = collections[0].name
    
    return Chroma(collection_name = collection_name, embedding_function = embeddings,
                  persist_directory = str(Path(persist_directory)),

                  )





