import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform

# Título de la aplicación
st.title('🌟 Generación Aumentada por Recuperación (RAG) 💬')

# Cargar y mostrar la nueva imagen
image = Image.open('nueva_imagen.png')  # Cambia el nombre del archivo a tu nueva imagen
st.image(image, width=350)

# Mostrar la versión de Python
st.write("🔍 Versión de Python:", platform.python_version())

# Sidebar para ingresar la clave de API
with st.sidebar:
    st.subheader("🔑 Configuración de API")
    ke = st.text_input('Ingresa tu Clave de API de OpenAI', type='password')

# Cargar el archivo PDF
pdf_file = st.file_uploader("📂 Carga el archivo PDF", type="pdf")

if pdf_file is not None:
    # Leer el PDF
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Dividir el texto en fragmentos
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)

    # Crear embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Pregunta del usuario
    st.subheader("❓ ¿Qué quieres saber sobre el documento?")
    user_question = st.text_area("Escribe tu pregunta aquí:")

    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        # Configurar el modelo de lenguaje
        llm = OpenAI(model_name="gpt-4o-mini")
        chain = load_qa_chain(llm, chain_type="stuff")

        # Obtener respuesta
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)

        # Mostrar respuesta
        st.write("📄 **Respuesta:**", response)
