from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List
from functions import (
    hash_password, verify_password, send_email, get_vectorstore, load_docs_from_file, conn, cur, agent, chat_llm, llm_with_tools, save_uploaded_file
)
from jwt_handler import verify_token
from datetime import datetime

import os
import tempfile
import shutil
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import TypedDict , Optional
from langgraph.prebuilt import ToolNode, tools_condition
from fastapi.responses import StreamingResponse
import json
from fastapi.responses import FileResponse
from mimetypes import guess_type
router = APIRouter()


llm_with_tools = chat_llm.bind_tools([send_email])

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
from pathlib import Path

@router.get("/assets/{file_path:path}")
async def serve_asset(file_path: str):
    """Serve uploaded files through API"""
    from functions import ASSETS_DIR
    from pathlib import Path
    from mimetypes import guess_type
    from urllib.parse import unquote

    # FastAPI decoded path, but be defensive: strip an accidental leading "assets/"
    if file_path.startswith("assets/"):
        file_path = file_path.split("assets/", 1)[1]

    # Also strip any leading slashes that somehow made their way here
    file_path = file_path.lstrip("/")

    full_path = os.path.join(ASSETS_DIR, file_path)

    # Security check
    if not os.path.exists(full_path):
        # helpful debug for server logs
        print(f"[serve_asset] NOT FOUND: {full_path} (requested path: {file_path})")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        resolved_path = Path(full_path).resolve()
        assets_path = Path(ASSETS_DIR).resolve()
        resolved_path.relative_to(assets_path)
    except ValueError:
        print(f"[serve_asset] ACCESS DENIED: {full_path}")
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine MIME type using the filename (works fine with full path as well)
    mime_type, _ = guess_type(full_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Logging so you can see asset requests in server logs
    print(f"[serve_asset] Serving: {full_path} (mime={mime_type})")

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }

    if mime_type.startswith("video/"):
        headers.update({
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        })

    return FileResponse(full_path, media_type=mime_type, headers=headers)


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

            # Persist uploaded file to assets/ and use that persistent path for ingestion
            saved_path = save_uploaded_file(tmp_path, uf.filename)

            chunks = load_docs_from_file(saved_path, ext, uf.filename)
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
    return {"message": f"Processed {sum(1 for r in results if r['status']=='success')} documents", "documents": results}



from fastapi import Request, BackgroundTasks as _BackgroundTasks, HTTPException
import json, os
from collections import defaultdict
from typing import List, Any

@router.post("/query")
async def query_documents(
    req: QueryRequest,
    request: Request,
    background_tasks: "BackgroundTasks" = None,
    current_user: str = Depends(verify_token)
):
    """
    Query endpoint (fixed):
     - similarity search with scores
     - group/merge segments per asset
     - create enriched sources (with absolute URLs and locators)
     - include locators in the LLM context (so LLM cites timestamps/pages)
     - stream the LLM answer; header X-Sources contains JSON of sources
    """
    from functions import SystemMessage, HumanMessage
    if background_tasks is None:
        background_tasks = _BackgroundTasks()

    if current_user != req.email:
        raise HTTPException(status_code=403, detail="Access denied")

    cur.execute("SELECT id, email FROM users WHERE email=%s", (req.email,))
    user = cur.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user_id, user_email = user

    q = (req.Query or "").strip()
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    # helper to build absolute URL for assets using request.base_url
    base_url = str(request.base_url).rstrip("/")  # e.g. "https://full-shrimp-deeply.ngrok-free.app"
    def make_absolute(u: str | None):
        if not u:
            return None
        if u.startswith("http://") or u.startswith("https://"):
            return u
        # allow either "/assets/..." or "assets/..."
        return base_url + (u if u.startswith("/") else f"/{u}")

    vs = get_vectorstore()
    refined = req.Query

    # Get hits (document + score)
    try:
        hits = vs.similarity_search_with_score(refined, k=5)  # list[(Document, score)]
    except Exception:
        docs = vs.similarity_search(refined, k=5)
        hits = [(d, 1.0) for d in docs]

    if not hits:
        return StreamingResponse(iter([json.dumps({"error": "No documents found"})]),
                                 media_type="application/json")

    # Group hits by asset URI (so multiple segments from same file collapse)
    asset_groups = defaultdict(list)
    for doc, score in hits:
        meta = doc.metadata or {}
        asset = meta.get("asset_uri") or meta.get("source") or meta.get("file_path") or "unknown_asset"
        try:
            start = float(meta.get("start")) if meta.get("start") is not None else None
            end = float(meta.get("end")) if meta.get("end") is not None else None
        except Exception:
            start, end = None, None

        asset_groups[asset].append({
            "doc": doc,
            "score": float(score) if score is not None else 0.0,
            "start": start,
            "end": end,
            "meta": meta
        })

    # Merge close/overlapping segments and pick representative
    def pick_and_merge_segments(segs, merge_gap_s=1.5):
        segs_with_time = [s for s in segs if s["start"] is not None and s["end"] is not None]
        if not segs_with_time:
            best = max(segs, key=lambda x: x["score"])
            return best.get("start"), best.get("end"), best.get("meta"), best.get("doc")

        segs_with_time.sort(key=lambda x: x["start"])
        seed = max(segs_with_time, key=lambda x: x["score"])
        merged_start, merged_end = seed["start"], seed["end"]

        # expand backward and forward to include nearby segments
        for s in reversed(segs_with_time):
            if s["end"] >= merged_start - merge_gap_s:
                merged_start = min(merged_start, s["start"])
                merged_end   = max(merged_end, s["end"])
        for s in segs_with_time:
            if s["start"] <= merged_end + merge_gap_s:
                merged_start = min(merged_start, s["start"])
                merged_end   = max(merged_end, s["end"])

        return float(merged_start), float(merged_end), seed.get("meta"), seed.get("doc")

    # Build final sources (one entry per distinct asset)
    final_sources = []
    for asset_uri, segs in asset_groups.items():
        start, end, rep_meta, rep_doc = pick_and_merge_segments(segs)
        meta = dict(rep_meta or {})
        # ensure asset_uri present in metadata
        meta["asset_uri"] = asset_uri

        # determine doc type
        dtype = meta.get("doc_type") or (asset_uri.split(".")[-1] if "." in asset_uri else None)
        if isinstance(dtype, str):
            dtype = dtype.lower()

        display = {}
        complete_source = {
            "source": meta.get("source") or os.path.basename(asset_uri),
            "asset_uri": meta.get("asset_uri")
        }

        # only attach start/end when available
        if start is not None and end is not None:
            display["start_time"] = round(start, 2)
            display["end_time"] = round(end, 2)
            display["time_range"] = f"{round(start,2)}s - {round(end,2)}s"
            complete_source["start"] = round(start,2)
            complete_source["end"]   = round(end,2)

        # choose display type and absolute URLs
        if dtype in ("mp4","mov","webm") or meta.get("doc_type") == "video":
            display["type"] = "video_player"
            complete_source["video_url"] = make_absolute(asset_uri)   # use asset_uri directly
            complete_source["transcript"] = meta.get("transcript")
        elif dtype in ("mp3","wav") or meta.get("doc_type") == "audio":
            display["type"] = "audio_player"
            complete_source["video_url"] = make_absolute(meta.get("asset_uri"))
            complete_source["transcript"] = meta.get("transcript")
        elif dtype == "pdf" or meta.get("doc_type") == "pdf":
            display["type"] = "pdf_page"
            # adopt common meta keys used when ingesting pages: 'page' or 'page_number'
            page_num = meta.get("page") or meta.get("page_number")
            if page_num is not None:
                complete_source["page_number"] = int(page_num)
            # full_text: extracted page text (if you stored it at ingestion)
            if "full_text" in meta:
                complete_source["full_text"] = meta.get("full_text")
            complete_source["pdf_url"] = make_absolute(meta.get("asset_uri"))
            display["locator"] = f"{complete_source['source']} (p.{complete_source.get('page_number','?')})"
        elif dtype == "csv" or meta.get("doc_type") == "csv":
            display["type"] = "csv_table"
            complete_source["csv_url"] = make_absolute(meta.get("asset_uri"))
            complete_source["row_number"] = meta.get("row") or meta.get("row_number")
            complete_source["headers"] = meta.get("csv_columns") or []
            complete_source["complete_row"] = meta.get("row_data") or {}
        elif meta.get("doc_type") == "docx":
            display["type"] = "docx_paragraph"
            complete_source["paragraph_index"] = meta.get("paragraph_index")
            complete_source["full_text"] = meta.get("doc_text") or meta.get("full_text")
        elif meta.get("doc_type") == "image" or asset_uri.lower().endswith((".png",".jpg",".jpeg")):
            display["type"] = "image_viewer"
            complete_source["image_url"] = make_absolute(meta.get("asset_uri"))
        else:
            display["type"] = "default"
            complete_source["text"] = rep_doc.page_content if rep_doc is not None else ""

        source_obj = {
            "search_chunk": rep_doc.page_content if rep_doc is not None else "",
            "metadata": meta,
            "display": display,
            "complete_source": complete_source
        }
        final_sources.append(source_obj)
        print(final_sources)

    # sort sources by highest seed score (descending)
    def seed_score(asset_uri):
        return max(s["score"] for s in asset_groups[asset_uri])
    final_sources.sort(key=lambda s: -seed_score(s["metadata"]["asset_uri"]))

    # Build the LLM context that includes locators (so model can cite time/page)
    def _locator_text(s):
        d = s.get("display", {})
        cs = s.get("complete_source", {})
        if d.get("type") == "video_player" and cs.get("start") is not None and cs.get("end") is not None:
            return f"{cs.get('source')} ({cs.get('start')}s-{cs.get('end')}s)"
        if d.get("type") == "pdf_page" and cs.get("page_number") is not None:
            return f"{cs.get('source')} (p.{cs.get('page_number')})"
        if d.get("type") == "csv_table" and cs.get("row_number") is not None:
            return f"{cs.get('source')} (row {cs.get('row_number')})"
        return cs.get("source")

    context_parts = []
    for i, s in enumerate(final_sources):
        locator = _locator_text(s)
        snippet = (s.get("search_chunk") or "").strip()
        context_parts.append(f"Doc {i+1} [{locator}]:\n{snippet}")

    context = "\n\n".join(context_parts)

    # System prompt includes explicit instruction to cite locators (file + time/page/row)
    system_msg = f"""
    You are a medical assistant. Answer ONLY using the provided CONTEXT below. 
    Each context item has a locator (file and page/time/row). When you cite a source, include the locator exactly as shown.
    Context:
    {context}
    """

    messages = [SystemMessage(content=system_msg), HumanMessage(content=f"Question: {q}")]

    # --- use your existing graph/llm tooling (unchanged) ---
    graph = StateGraph(MessagesState)
    def agent_node(state): return {"messages": [llm_with_tools.invoke(state["messages"])]}
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([send_email]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    builder = graph.compile()
    stream = builder.stream({"messages": messages})

    # streaming generator and DB save as before
    ai_chunks: List[str] = []
    def give_streaming_response():
        for event in stream:
            if "agent" in event and "messages" in event["agent"]:
                message = event["agent"]["messages"][-1]
                if hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
                    for tool_call in message.additional_kwargs["tool_calls"]:
                        try:
                            args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                            body = args.get("body", "")
                            if body:
                                yield body
                                ai_chunks.append(body)
                        except Exception:
                            continue
                elif getattr(message, "content", None):
                    yield message.content
                    ai_chunks.append(message.content)

    def _save_chat():
        try:
            full_output = "".join(ai_chunks)
            cur.execute(
                "INSERT INTO chat_threads (chatroom_id, user_message, ai_response) VALUES "
                "((SELECT id FROM chatrooms WHERE user_id=%s ORDER BY created_at DESC LIMIT 1), %s, %s)",
                (user_id, q, full_output),
            )
            conn.commit()
        except Exception as e:
            print("DB save failed:", e)

    background_tasks.add_task(_save_chat)

    # Return streaming response and attach the enriched sources (already absolute URLs)
    return StreamingResponse(
        give_streaming_response(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Sources": json.dumps(final_sources)}
    )
