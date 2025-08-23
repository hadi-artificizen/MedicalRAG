from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List
from functions import (
    hash_password, verify_password, send_email, get_vectorstore, 
    contains_prescription, load_docs_from_file, conn, cur
)
from jwt_handler import verify_token
from datetime import datetime
import os
import tempfile
import shutil

router = APIRouter()

class RegisterRequest(BaseModel):
    email: str
    password: str

class QueryRequest(BaseModel):
    email: str
    chatroom_id: int
    Query: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register")
def register(req: RegisterRequest):
    cur.execute("SELECT * FROM users WHERE email=%s", (req.email,))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="email already exists")
    
    hashed_password = hash_password(req.password)
    cur.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (
        req.email, hashed_password))
    conn.commit()
    return {"message": "User registered successfully"}

@router.post("/login")
def login(req: LoginRequest):
    cur.execute("SELECT email, password FROM users WHERE email=%s", (req.email,))
    user = cur.fetchone()
    
    if not user or not verify_password(req.password, user[1]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    from jwt_handler import create_access_token
    access_token = create_access_token(data={"sub": req.email})
    return {
        "message": "Login successful",
        "access_token": access_token,  
        "token_type": "bearer",       
        "user": {"email": req.email}
    }

@router.get("/chatrooms/{email}")
def get_chatrooms(email: str, current_user: str = Depends(verify_token)):
    if current_user != email:
        raise HTTPException(status_code=403, detail="Access denied")
    
    cur.execute("SELECT id FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user[0]
    cur.execute("SELECT id, created_at FROM chatrooms WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
    return {"chatrooms": cur.fetchall()}

@router.get("/messages/{chatroom_id}")
def get_messages(chatroom_id: int, current_user: str = Depends(verify_token)):
    cur.execute("""
        SELECT c.id FROM chatrooms c 
        JOIN users u ON c.user_id = u.id 
        WHERE c.id = %s AND u.email = %s
    """, (chatroom_id, current_user))
    
    if not cur.fetchone():
        raise HTTPException(status_code=403, detail="Access denied to this chatroom")
    
    cur.execute("SELECT user_message, ai_response FROM chat_threads WHERE chatroom_id=%s", (chatroom_id,))
    return {"messages": cur.fetchall()}

@router.post("/new_chatroom/{email}")
def new_chatroom(email: str, current_user: str = Depends(verify_token)):
    if current_user != email:
        raise HTTPException(status_code=403, detail="Access denied")
    
    cur.execute("SELECT id FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = user[0]
    cur.execute("INSERT INTO chatrooms (user_id, created_at) VALUES (%s, %s) RETURNING id", (user_id, datetime.utcnow()))
    chatroom_id = cur.fetchone()[0]
    conn.commit()
    return {"chatroom_id": chatroom_id}

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), current_user: str = Depends(verify_token)):
    
    if not files:
        raise HTTPException(400, "No files provided")
    vs, results = get_vectorstore(), []
    for uf in files:
        ext = os.path.splitext(uf.filename.lower())[1]
        if ext not in [".pdf", ".docx", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3"]:
            results.append({"filename": uf.filename, "status": "unsupported", "chunks_count": 0})
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(uf.file, tmp)
                tmp_path = tmp.name
            chunks = load_docs_from_file(tmp_path, ext, uf.filename)
            if not chunks:
                results.append({"filename": uf.filename, "status": "no content", "chunks_count": 0})
                continue
            vs.add_documents(chunks)
            results.append({"filename": uf.filename, "status": "success", "chunks_count": len(chunks)})
        except Exception as e:
            results.append({"filename": uf.filename, "status": str(e), "chunks_count": 0})
        finally:
            tmp_path and os.path.exists(tmp_path) and os.unlink(tmp_path)
    vs.persist()
    return {"message": f"Processed {sum(r['status']=='success' for r in results)} documents", "documents": results}

@router.post("/query")
async def query_documents(req: QueryRequest, current_user: str = Depends(verify_token)):
    from functions import chat_llm, SystemMessage, HumanMessage
    if current_user != req.email:
        raise HTTPException(status_code=403, detail="Access denied")
    
    cur.execute("SELECT id, email FROM users WHERE email=%s", (req.email,))
    user = cur.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user_id, user_email = user

    q = (req.Query or "").strip()
    if len(q) < 3:
        raise HTTPException(400, "Query too short")

    vs = get_vectorstore()
    refined = (req.Query)
    docs = vs.similarity_search(refined, k=5)
    if not docs:
        return {"answer": "No relevant info.", "sources": []}

    context = "\n\n".join(f"Doc {i+1}:\n{d.page_content}" for i, d in enumerate(docs))
    system_msg = ("You are a helpful medical assistant. "
                  "If you recommend medicine, format as 'Prescription: <name and dosage>'. "
                  "Answer based ONLY on context. If unknown, say so.")
    messages = [SystemMessage(content=system_msg), HumanMessage(content=f"Context:\n{context}\n\nQuestion: {q}")]
    answer = chat_llm.invoke(messages).content

    cur.execute("INSERT INTO chat_threads (chatroom_id, user_message, ai_response) VALUES "
                "((SELECT id FROM chatrooms WHERE user_id=%s ORDER BY created_at DESC LIMIT 1), %s, %s)", (user_id, q, answer))
    conn.commit()

    prescription_detected = contains_prescription(answer)
    email_sent = False
    if prescription_detected:
        email_sent = send_email(user_email, "Medical Query Response", f"Query: {q}\n\nAnswer:\n{answer}")

    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    return {"answer": answer, "sources": sources, "email_sent": email_sent}