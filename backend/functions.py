import os
import re
from datetime import datetime
from typing import List
import psycopg2
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
from pydub import AudioSegment
from langchain.tools import tool, StructuredTool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field
from typing import TypedDict, Optional

load_dotenv()

DB_URL = os.getenv("DB_URL")
conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)

chat_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    base_url="https://api.groq.com/openai/v1"
)

SMTP_USER = "imabdul.hadi1234@gmail.com"
SMTP_PASSWORD = "gqty jnji ufjq rgrh"
EMAIL_TIMEOUT = 30

PERSIST_DIR = "D:/LangchainPractice/RAG/chroma_db"

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def send_email(to_email , subject , body) :

    """Send an email to the given address with subject and body. Call this function for ANY medical query response."""
    print("Email function called")
    if not to_email or "@" not in to_email:
        return {"success": False, "message": "Invalid email address"}
    
    if not body.strip():
        return {"success": False, "message": "Email body cannot be empty"}
    
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=EMAIL_TIMEOUT) as server:
            print("SMTP connection established")
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully to {to_email}")
        return {"messages" : f"success : True, message : Email sent to {to_email}"}
    except Exception as e:
        error_msg = f"Email failed: {str(e)}"
        print(error_msg)
        return {"messages" : f"success : False, message : {error_msg}"}

class SendEmailInput(BaseModel):
    to_email: str = Field(default="", description="Recipient email address")
    subject: str = Field(default="Medical Query Response", description="Subject of the email")
    body: str = Field(..., description="Body of the email containing the medical response")

send_email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="Send an email with the medical response. ALWAYS call this function after providing any medical advice or response.",
    args_schema=SendEmailInput
)

# Initialize agent with explicit instructions
from langgraph.graph import StateGraph, END

# Bind LLM with the send_email tool
llm_with_tools = chat_llm.bind_tools([send_email_tool])

class graphState(TypedDict):
    query : str
    doc : str
    mail : Optional[str]
    email_body : Optional[str]
    subject : Optional[str]
# Define graph state (simple dict is fine)
graph = StateGraph(graphState)

def agent_node(state):

    query = state["query"]
    doc = state["doc"]



    # Let LLM respond
    # response = llm_with_tools.invoke(state["messages"])
    # state["messages"].append(response)

    # # Check if the model requested a tool
    # tool_calls = getattr(response, "tool_calls", None)
    # if tool_calls:
    #     for tc in tool_calls:
    #         if tc["name"] == "send_email":
    #             try:
    #                 # Parse args
    #                 args = tc["args"]
    #                 result = send_email(**args)
    #                 # Append tool result back into the conversation
    #                 state["messages"].append(
    #                     {"role": "tool", "name": "send_email", "content": str(result)}
    #                 )
    #             except Exception as e:
    #                 state["messages"].append(
    #                     {"role": "tool", "name": "send_email", "content": f"Error: {str(e)}"}
    #                 )
    # return state


graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

# Compile graph
agent = graph.compile()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="abhinand/MedEmbed-base-v0.1",
        model_kwargs={"device": "cpu"}
    )

def get_vectorstore():
    return Chroma(embedding_function=get_embeddings(), persist_directory=PERSIST_DIR)


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
    elif ext in [".mp4", ".mp3"]:
        try:
            print(f"Processing audio/video file: {source}")
            base, _ = os.path.splitext(tmp_path)
            audio_path = base + "_audio.wav"
            if ext == ".mp4":
                video = VideoFileClip(tmp_path)
                audio_path = tmp_path.replace(".mp4", "_audio.wav")
                video.audio.write_audiofile(audio_path)
                video.close()
            
            elif ext == ".mp3":
                audio = AudioSegment.from_file(tmp_path, format="mp3")  
                audio_path = tmp_path.replace(".mp3", "_audio.wav")
                audio.export(audio_path, format="wav")               

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