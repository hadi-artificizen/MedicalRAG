import os
import re
from datetime import datetime
from typing import List
import psycopg2
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from docx import Document as DocxDocument
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import bcrypt
import pytesseract
from PIL import Image
import whisper
from moviepy.editor import VideoFileClip

load_dotenv()


DB_URL = os.getenv("DB_URL")
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


chat_llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    base_url="https://api.groq.com/openai/v1"
)


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "imabdul.hadi1234@gmail.com"
SMTP_PASSWORD = "gqty jnji ufjq rgrh"
EMAIL_TIMEOUT = 30


PERSIST_DIR = "D:/LangchainPractice/RAG/chroma_db"

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def send_email(to_email: str, subject: str, body: str) -> bool:
    if not to_email or "@" not in to_email:
        return False
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="abhinand/MedEmbed-base-v0.1",
        model_kwargs={"device": "cpu"}
    )

def get_vectorstore():
    return Chroma(embeddings_functions=get_embeddings(), persist_directory=PERSIST_DIR)

def contains_prescription(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in [
        "Prescription:", "prescribed?", "take.*mg", "dosage", "medicine",
        "medication", "drug", "tablet", "capsule", "syrup", "inject",
        "twice.*day", "once.*day", "three.*times"
    ])

def refine_query(original_query: str) -> str:
    prompt = ("You are a query rewriting assistant for a medical retrieval system. "
              "Rewrite the user's query clearly. If vague, expand. If single-word, "
              "make it a medical question. Keep only the query.")
    resp = chat_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Original query: {original_query}")
    ])
    return resp.content.strip()

def load_docs_from_file(tmp_path: str, ext: str, source: str) -> List[Document]:
    meta = {"source": source, "processed_date": datetime.now().isoformat()}
    docs = []
    if ext == ".pdf":
        docs = [d for d in PyPDFLoader(tmp_path).load()]
        [d.metadata.update(meta) for d in docs]
    elif ext == ".docx":
        docs = [Document(page_content="\n".join(p.text for p in DocxDocument(tmp_path).paragraphs), metadata=meta)]
    elif ext == ".csv":
        docs = [d for d in CSVLoader(tmp_path, encoding="utf-8").load()]
        [d.metadata.update(meta) for d in docs]
    elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
        try:
            img = Image.open(tmp_path)
            text = pytesseract.image_to_string(img)
            if text.strip():
                docs = [Document(page_content=text, metadata=meta)]
            else:
                return []
        except Exception as e:
            print(f"OCR failed for {source}: {e}")
            return []
    elif ext == ".mp4":
        try:
            
            video = VideoFileClip(tmp_path)
            audio_path = tmp_path.replace(".mp4", "_audio.wav")
            video.audio.write_audiofile(audio_path)
            video.close() 

            model = whisper.load_model("base") 
            transcription = model.transcribe(audio_path)
            text = transcription["text"]
            
           
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            if text.strip():
                docs = [Document(page_content=text, metadata=meta)]
            else:
                return []
        except Exception as e:
            print(f"Transcription failed for {source}: {e}")
            return []
    return splitter.split_documents(docs)
    