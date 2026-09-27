from functools import lru_cache # least recently used cache- it is a built in decorator
from langchain_huggingface import HuggingFaceEmbeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
@lru_cache(maxsize=1)
def get_embeddings()-> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL, 
        model_kwargs = {"device":"cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

