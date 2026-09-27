import os
from pathlib import Path
import streamlit as st
app_title = "mypdfprocessor"
chroma_dir = str(Path("chroma_db").resolve())
default_model = "gemini-3.6-flash"
def get_secret(name:str, default:str|None=None)-> str|None:
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)
    
    

