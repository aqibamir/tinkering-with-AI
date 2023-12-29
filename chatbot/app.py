import sys
import os
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import streamlit as st

load_dotenv()

# Prompt templates 
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
                                                        
                                                        
Given the following conversation and a follow up question by the user. You need to creat a standalon question that enhances the user question by including keywords that would
help a search algorithm find the relevant documents using semantic search. The grammatical structure of the question does not matter.
                                                        
==== FOLLOW UP QUESTION ===
    {question}
========================
                                                        
Standalone question:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template("""You are an assistant who is answering user questions about some books. Some excrepts from the book
 are given as ```context``` below. You have to now answer the user question the question based only on this information. The response should not include sentences
 like 'Based on the context' unless you can not infer the user questions answer from the context.   

=== CHAT HISTORY ===
    {chat_history}

====================
                                                 
    === CONTEXT ===
       {context}
    ===============

USER QUESTIONN: {question}
""")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def process_pdf(filename):
    """ Processes a PDF file and returns its content in chunks """
    
    text = ""
    pdf_reader = PdfReader(filename)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    metadata = [{"book_name": filename} for i in range(len(text))]
    chunks = text_splitter.create_documents(texts=[text],metadatas=metadata)

    return chunks

def get_vectorstore():
    persist_directory = './db'
    embeddings = OpenAIEmbeddings()
    chunks = []
    if not os.path.exists(persist_directory):
        crime_and_punishment = process_pdf('./books/cap.pdf')
        alchemist = process_pdf('./books/alchemist.pdf')
        chunks = crime_and_punishment + alchemist
        vectorstore = Chroma.from_documents(embedding = embeddings, documents = chunks,persist_directory = persist_directory)
        return vectorstore
    
    vectorstore = Chroma(embedding_function = embeddings, persist_directory = persist_directory)

    return vectorstore

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    """ Combines a list of documents into a single formatted string """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def handle_user_input(user_question,memory,retriever,llm):
    """Generates a response to a user question based on chat history and retrieved documents.

    Args:
        user_question (str): User's question.
        memory (ConversationBufferMemory): Conversation buffer memory.
        retriever: Document retriever.
        llm: Large language model.

    Returns:
        str: AI generated response.
    """

    # Fetching chat history
    past_messages = get_buffer_string(memory.load_memory_variables({})['history'])

    # Generating a standalone question to retrieve relevant documents
    standalone_question_chain = CONDENSE_QUESTION_PROMPT | ChatOpenAI(model = 'gpt-3.5-turbo-1106',temperature = 0) | StrOutputParser()
    standalone_question = standalone_question_chain.invoke({"chat_history":past_messages,"question":user_question},config = {'callbacks': [ConsoleCallbackHandler()]})
    
    # Chain to fetch documents for in memory context
    get_documents_chain = retriever | _combine_documents
    context = get_documents_chain.invoke(standalone_question)

    # Chain to generate final answer based on context , user question and chat history
    final_answer_chain = ANSWER_PROMPT | ChatOpenAI(model = 'gpt-3.5-turbo-1106',temperature = 0) | StrOutputParser()
    response = final_answer_chain.invoke({"context": context, "question": user_question,"chat_history":past_messages},config = {'callbacks': [ConsoleCallbackHandler()]}) 

    # Saving the response in memory
    memory.save_context({'question':user_question}, {'answer':response})
    
    return response

def main():
    st.set_page_config(page_title="Chat PDF BOOK", page_icon=":books:")

    st.header("Chat with 'The Alchemist' and 'Crime and Punishemt' :books:")

    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = get_vectorstore()
        st.session_state['retriever'] = st.session_state['vector_store'].as_retriever(search_kwargs={"k": 5})
        st.session_state['memory'] = ConversationBufferMemory( memory_key='history', return_messages=True, output_key="answer", input_key="question")

    msgs = StreamlitChatMessageHistory(key="chat_history")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Get the chat response 
        response = handle_user_input(prompt , st.session_state['memory'], st.session_state['retriever'], OpenAI())

        st.chat_message("ai").write(response)
        msgs.add_ai_message(response)

if __name__ == "__main__":
    main()

