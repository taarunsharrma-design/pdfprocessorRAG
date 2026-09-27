from pathlib import Path 
from tempfile import NamedTemporaryFile
from typing import Iterable
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdfs(uploaded_files:Iterable, chunk_size:int=1000, chunk_overlap:int=200)->list[Document]:
    if(chunk_overlap>=chunk_size):
        raise ValueError("chunkoverlap must be smaller than chunksize")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    all_chunks:list[Document]=[] # creating Empty list to store all the processed document chunk across all files

    for uploaded_file in uploaded_files: # Processing uploaded files
        suffix = Path(uploaded_file.name).suffix or ".pdf"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file: # creating a temporary file on disk with extracted suffix ".pdf" , delete false mentioned to ensure file exists even after closing
            temp_file.write(uploaded_file.getvalue()) # reading rawbyte content of the uploaded and writing in our temp files on disk
            temp_Path = temp_file.name # saving temp file in a variable

        try:
            loader = PyPDFLoader(temp_Path)   
            pages = loader.load() # reading and parsing the pdf and returning the list of document object one page as one document object
            for page_doc in pages:
                page_doc.metadata["source"]= uploaded_file.name # replacing temp file Path in metadata source with original uploaded file name
                page_doc.metadata["filename"]= uploaded_file.name # adding explicit file name key to metadata to the original file name
                page_doc.metadata["page"]= page_doc.metadata.get("page", 0,) # accessing page key in metadata then retrieving existing page index; setting default to zero if key not present

            chunks = splitter.split_documents(pages)
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            all_chunks.extend(chunks)

        finally:
            Path(temp_Path).unlink(missing_ok= True)

    return all_chunks


                

