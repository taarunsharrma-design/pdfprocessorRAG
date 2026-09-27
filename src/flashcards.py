from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def _generate(rag_engine, topic:str, instruction:str,)-> Dict[str, Any]:
    documents = rag_engine.retrieve(topic, top_k =7)
    if not documents:
        return{
            "answer":"no relevant material was found",
            "sources": []
        }
    context = rag_engine.format_context(documents)
    prompt = ChatPromptTemplate.from_template("""
    You are an university assistant
    Use only the supplied pdfs and context
    {instruction}
    Topic:
    {topic}
    Context:
    {context}
    requirements:
    Do not invent facts
    Keep the material faithful to the uploaded documents
    Include source citation in the form [Source:filename, Page X]




     """)
    chain = prompt|rag_engine.llm|StrOutputParser()
    answer = chain.invoke({
        "instruction":instruction,
        "topic": topic,
        "context": context,
    })
    return{
                "answer":answer,
                "sources": rag_engine.extract_sources(documents),
            }

def generate_summary(rag_engine, topic:str)->Dict[str, Any]:
    return _generate(rag_engine, topic, """

    create a structured revision summary which includes
    1. Key points
    2. main process
    3. Any argument or example within the document

     """,)

def generate_flashcards(rag_engine, topic:str)->Dict[str, Any]:
    return _generate(rag_engine, topic, """ 
    create 10 revision flashcards
    format every card as:
    ### Card 1
    **Q:Question**
    **A: Answer**
    Question should consider important key points from the pdf

    """,)
