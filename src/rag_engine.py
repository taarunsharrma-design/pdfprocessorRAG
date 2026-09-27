from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from src.config import default_model

class RAGEngine:
    def __init__(self,vector_store, api_key:str, model_name:str= default_model, temperature:float=0.2, top_k:int=5,  ):
        self.vector_store =vector_store
        self.top_k = top_k
        self.llm = ChatGoogleGenerativeAI(model = model_name, google_api_key = api_key, temperature= temperature)
        self.prompt = ChatPromptTemplate.from_template("""

        You are an academic assistant helping a student understand
        their uploaded course materials.

        Answer the user's question using ONLY the provided context.

        Rules:
        1. Do not invent information.
        2. If the answer cannot be found in the context, say:
        "I couldn't find this information in the uploaded documents."
        3. Explain concepts clearly and accurately.
        4. Prefer concise but useful answers.
        5. Cite claims using [Source: filename, Page X].
        6. Do not cite sources that are not present in the context.

        Context:
        ----------------
        {context}
        ----------------

        Question:
        {question}

        Answer:

        """)

        self.chain = self.prompt | self.llm | StrOutputParser() # this is LCEL Langchain expression language constructing

    def retrieve(self, question: str, top_k:Optional[int]=None, )->List[Document]: #Helper function to run queries on vector db
        k = top_k or self.top_k
        return self.vector_store.similarity_search(question, k=k, )

    @staticmethod 
    def format_context(documents:List[Document])->str:
        context_parts= []
        for i,doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            source = (metadata.get("filename") or metadata.get("source") or "unknown document")
            page = metadata.get("page")
            if isinstance(page, int):
                page_number = page+1
            else:
                page_number = page or "unknownn"

            context_parts.append(f"""[Document{i}] source:{source} page:{page_number} content:{doc.page_content}""")
        return "\n".join(context_parts)

    
    @staticmethod 
    def extract_sources(documents:List[Document],)->str:
        sources = []
        seen = set()
        for doc in documents:
            metadata = doc.metadata or {}
            source = (metadata.get("filename") or metadata.get("source") or "unknown document")
            page = metadata.get("page")
            if isinstance(page, int):
                page = page+1

            key = (source, page)

            if key in seen:
                continue

            seen.add(key)
            sources.append(
                {
                  "filename":source,
                  "page": page

                }
            )
        return sources


    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "documents": [],
            }

        documents = self.retrieve(
            question.strip(),
            top_k=top_k,
        )

        if not documents:
            return {
                "answer": (
                    "I couldn't find any relevant information "
                    "in the uploaded documents."
                ),
                "sources": [],
                "documents": [],
            }

        context = self.format_context(documents)

        answer = self.chain.invoke(
            {
                "context": context,
                "question": question.strip(),
            }
        )

        return {
            "answer": answer,
            "sources": self.extract_sources(documents),
            "documents": documents,
        }

    def similarity_search_with_scores(
        self,
        question: str,
        top_k: Optional[int] = None,
    ):
        k = top_k or self.top_k

        return self.vector_store.similarity_search_with_score(
            question,
            k=k,
        )

       


        
    

