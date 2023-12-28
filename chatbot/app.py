
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import OpenAI

# Creates chunks of the PDF file
def process_pdf(filename):
    text = ""
    pdf_reader = PdfReader(filename)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )

    metadata = [{"book_name": filename} for i in range(len(text))]
    chunks = text_splitter.create_documents(texts=[text],metadatas=metadata)

    return chunks


def get_vectorstore():
    embeddings = OpenAIEmbeddings()

    crime_and_punishment = process_pdf('cap.pdf')
    alchemist = process_pdf('alchemist.pdf')

    vectorstore = Chroma.from_documents(embedding=embeddings, documents= alchemist+ crime_and_punishment)

    return vectorstore


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

 === CHAT HISTORY ===
       {chat_history}

 ====================
                                                        
==== Follow Up Input ===
    {question}
========================
                                                        
Standalone question:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template("""You are an assistant who is answering user questions about some books. Some excrepts from the book
 are given as ```context``` below. Also the previous chat history with the user is also given. You have to now answer the user question the question based only on this information.

    === CHAT HISTORY ===
       {chat_history}

    ====================

    === CONTEXT ===
       {context}
    ===============

USER QUESTIONN: {question}
""")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def get_response(user_question,memory,retriever,llm):
    # First create the chain to condense the chat history

    # Fetching the chat history
    past_messages = get_buffer_string(memory.load_memory_variables({})['history'])

    # Generating a standalone question to retrieve relevant documents
    standalone_question_chain = CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser()
    standalone_question = standalone_question_chain.invoke({"chat_history":past_messages,"question":user_question})
    
    # Chain to fetch documents for in memory context
    get_documents_chain = retriever | _combine_documents
    context = get_documents_chain.invoke(standalone_question)

    print(context)

    # Chain to generate final answer based on context , user question and chat history
    final_answer_chain = ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
    response = final_answer_chain.invoke({"context": context, "question": user_question,"chat_history":past_messages}) 

    # Saving the response
    memory.save_context({'question':user_question}, {'answer':response})
    
    return response


    
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF BOOK", page_icon=":books:")

    st.header("Chat with 'The Alchemist' and 'Crime and Punishemt' :books:")

    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = get_vectorstore()
        st.session_state['retriever'] = st.session_state['vector_store'].as_retriever()
        st.session_state['memory'] = ConversationBufferMemory( memory_key='history', return_messages=True, output_key="answer", input_key="question")

    msgs = StreamlitChatMessageHistory(key="chat_history")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Get the chat response 
        response = get_response(prompt , st.session_state['memory'], st.session_state['retriever'], OpenAI())

        st.chat_message("ai").write(response)
        msgs.add_ai_message(response)
main()

