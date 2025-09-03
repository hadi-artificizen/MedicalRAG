import json
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from io import StringIO, BytesIO
import os

import json
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from io import StringIO, BytesIO
import os

# -------------------------
# Config
# -------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
HEADERS_BASE = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "StreamlitApp/1.0"  # Add user agent
}
st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")

# -------------------------
# Helpers
# -------------------------
def normalize_url(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{BACKEND_URL}/{path.lstrip('/')}"

def authed_headers() -> Dict[str, str]:
    headers = dict(HEADERS_BASE)
    headers["ngrok-skip-browser-warning"] = "true"  # Add this for all requests
    token = st.session_state.get("access_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def read_sources_header(resp: requests.Response) -> List[Dict[str, Any]]:
    """Parse the X-Sources header from backend and normalize asset urls"""
    try:
        raw = resp.headers.get("X-Sources")
        if not raw:
            return []
        sources = json.loads(raw)
        for s in sources:
            cs = s.get("complete_source") or {}
            for key in ("asset_uri", "video_url", "image_url", "pdf_url", "csv_url"):
                if cs.get(key):
                    cs[key] = normalize_url(cs[key])
            s["complete_source"] = cs
        return sources
    except Exception as e:
        st.error(f"Error parsing sources: {e}")
        return []

# -------------------------
# Display functions
# -------------------------
import streamlit as st
import requests
from io import BytesIO

def display_pdf_page(pdf_url: str, page_number: int | None, source_name: str):
    st.subheader(f"📄 PDF Source: {source_name}")
    
    # Add ngrok bypass parameter
    pdf_url_with_bypass = f"{pdf_url}?ngrok-skip-browser-warning=true"
    
    try:
        if page_number is not None:
            human_page = page_number
            st.markdown(f"**Page {human_page}**")
        
        # Use object tag instead of iframe for better PDF support
        st.markdown(f"""
        <object data="{pdf_url_with_bypass}" type="application/pdf" width="100%" height="600px">
            <embed src="{pdf_url_with_bypass}" type="application/pdf" width="100%" height="600px" />
            <p>Unable to display PDF. <a href="{pdf_url}" target="_blank">Click here to open in new tab</a></p>
        </object>
        """, unsafe_allow_html=True)
        
        # Fetch for download button
        headers = {"ngrok-skip-browser-warning": "true"}
        r = requests.get(pdf_url, headers=headers, timeout=20)
        r.raise_for_status()
        st.download_button("⬇️ Download PDF", r.content, file_name=source_name, mime_type="application/pdf")
        
    except Exception as e:
        st.error(f"Could not load PDF: {e}")
        st.markdown(f"[📎 Open PDF in New Tab]({pdf_url})")

# def display_pdf_page(pdf_url: str, page_number: int | None, source_name: str):
#     st.subheader(f"📄 PDF Source: {source_name}")
#     pdf_url_with_bypass = f"{pdf_url}?ngrok-skip-browser-warning=true"
#     try:
#         # Fetch PDF file (if it’s stored locally or in backend assets)
#         r = requests.get(pdf_url, timeout=20)
#         r.raise_for_status()

#         pdf_bytes = BytesIO(r.content)

#         if page_number is not None:
#             human_page = page_number + 1
#             iframe_url = f"{pdf_url}#page={human_page}"
#             st.markdown(f"**Opening at Page {human_page}:**", unsafe_allow_html=True)
#         else:
#             iframe_url = pdf_url

#         # Embed PDF in an iframe
#         st.markdown(
#             f'<iframe src="{iframe_url}" width="100%" height="600"></iframe>',
#             unsafe_allow_html=True
#         )

#         # Provide download option
#         st.download_button("⬇️ Download PDF", pdf_bytes, file_name=source_name)

#     except Exception as e:
#         st.error(f"Could not load PDF: {e}")
#         st.markdown(f"[Open PDF Directly]({pdf_url})")



import csv
import io

def display_csv_row(csv_url, headers, complete_row, row_number, title="CSV Row"):
    st.subheader(f"📊 {title}")
    st.write(f"**File:** {csv_url.split('/')[-1]}")
    st.write(f"**Row {row_number}**")

    try:
        if csv_url.startswith("http"):
            # Fetch from URL
            response = requests.get(csv_url)
            response.raise_for_status()
            content = response.content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(content))
        else:
            # Open local file
            with open(csv_url, "r", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)

        # Find the right row
        for idx, row in enumerate(reader, start=1):
            if idx == int(row_number):
                st.table([row])  # ✅ show only that row
                break
    except Exception as e:
        st.error(f"CSV display error: {e}")


def display_docx_preview(docx_url: str, source_name: str, paragraph_index: int | None, full_text: str | None):
    """Show DOCX paragraph with download option"""
    st.subheader(f"📝 DOCX Source: {source_name}")

    if full_text:
        st.markdown(f"""
        <div style="border-left: 4px solid #2E86AB; padding: 16px; background: #f8f9fa; margin: 10px 0; border-radius: 4px;">
            <p style="margin: 0; line-height: 1.6;">{full_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # Download button
    try:
        headers = {"ngrok-skip-browser-warning": "true"}
        r = requests.get(docx_url, headers=headers, timeout=20)
        if r.ok:
            st.download_button(
                "⬇️ Download DOCX", 
                r.content, 
                file_name=source_name,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.markdown(f"[📎 View DOCX]({docx_url})")
    except Exception as e:
        st.warning(f"Could not fetch DOCX: {e}")
        st.markdown(f"[📎 View DOCX]({docx_url})")


from streamlit.components.v1 import html
def display_video_segment(video_url, start, end, transcript=None, title="Video", key="0", autoplay=False):
    st.subheader(f"🎥 Video Source: {title}")
    
    # Add ngrok headers to video requests
    video_with_headers = f"{video_url}?ngrok-skip-browser-warning=true"
    
    auto_attrs = "autoplay muted" if autoplay else ""
    html(f"""
      <style>
        video {{ 
          border-radius: 8px; 
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
      </style>
      <video id="vid_{key}" width="100%" height="400" controls preload="metadata" playsinline {auto_attrs} crossorigin="anonymous">
        <source src="{video_with_headers}" type="video/mp4">
        <p>Your browser does not support the video tag. <a href="{video_url}" target="_blank">Click here to view the video</a></p>
      </video>
      <script>
        (function() {{
          const v = document.getElementById("vid_{key}");
          const START = {float(start)};
          const END   = {float(end)};
          
          v.addEventListener("loadedmetadata", function() {{
            v.currentTime = START;
          }});
          
          v.addEventListener("timeupdate", function() {{
            if (v.currentTime >= END) {{
              v.pause();
              v.currentTime = START;
            }}
          }});
          
          // Handle errors
          v.addEventListener("error", function(e) {{
            console.error("Video error:", e);
            v.style.display = "none";
            const errorMsg = document.createElement("div");
            errorMsg.innerHTML = '<p style="color: red;">Error loading video. <a href="{video_url}" target="_blank">Click to open directly</a></p>';
            v.parentNode.insertBefore(errorMsg, v.nextSibling);
          }});
        }})();
      </script>
    """, height=460, scrolling=False)

    if transcript:
        st.caption(f"📝 Transcript: {transcript}")
# def display_video_segment(video_url, start, end, transcript=None, title="Video", key="0", autoplay=False):
#     st.subheader(f"🎥 Video Source: {title}")
#     # NOTE: autoplay on modern browsers requires muted
#     auto_attrs = "autoplay muted" if autoplay else ""
#     html(f"""
#       <video id="vid_{key}" width="100%" height="400" controls preload="metadata" playsinline {auto_attrs}>
#         <!-- #t hint encourages the browser to start near 'start' -->
#         <source src="{video_url}#t={float(start)},{float(end)}" type="video/mp4">
#         Your browser does not support the video tag.
#       </video>
#       <script>
#         (function() {{
#           const v = document.getElementById("vid_{key}");
#           const START = {float(start)};
#           const END   = {float(end)};
#           const NUDGE = 0.001; // nudge to avoid stuck seeks at exact boundary

#           function clamp() {{
#             if (v.currentTime < START) v.currentTime = START;
#             if (v.currentTime >= END) {{
#               v.pause();
#               v.currentTime = START; // loop-like behavior on replay
#             }}
#           }}

#           // Some browsers ignore currentTime until ready; cover multiple stages
#           function seekToStart() {{
#             try {{ v.currentTime = START + NUDGE; }} catch (e) {{}}
#           }}

#           v.addEventListener("loadedmetadata", seekToStart, {{ once: true }});
#           v.addEventListener("canplay", seekToStart, {{ once: true }});
#           v.addEventListener("loadeddata", seekToStart, {{ once: true }});

#           // Enforce boundaries during playback
#           v.addEventListener("timeupdate", clamp);

#           // Block scrubbing outside the window
#           v.addEventListener("seeking", function() {{
#             if (v.currentTime < START) v.currentTime = START;
#             if (v.currentTime > END) v.currentTime = END - NUDGE;
#           }});

#           // If autoplay requested but blocked, try to play after seeking
#           { "v.play().catch(()=>{});" if autoplay else "" }
#         }})();
#       </script>
#     """, height=440, scrolling=False)

#     if transcript:
#       st.caption(f"📝 Transcript: {transcript}")
      
def display_image_with_ocr(image_url: str, ocr_text: str | None, source_name: str):
    """Show image with OCR text"""
    st.subheader(f"🖼️ Image Source: {source_name}")
    
    try:
        # Add headers for image requests
        headers = {"ngrok-skip-browser-warning": "true"}
        
        # Display image with error handling
        st.markdown(f"""
        <img src="{image_url}?ngrok-skip-browser-warning=true" 
             style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
        <div style="display: none; padding: 20px; background: #f0f0f0; text-align: center; border-radius: 8px;">
            <p>Image could not be loaded. <a href="{image_url}" target="_blank">Click to view directly</a></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Could not load image: {e}")
        st.markdown(f"[🖼️ View Image]({image_url})")
    
    if ocr_text and ocr_text.strip():
        with st.expander("🔍 OCR Text"):
            st.text_area("Extracted Text", ocr_text, height=200, key=f"ocr_{hash(image_url)}")
            
# def display_image_with_ocr(image_url: str, ocr_text: str | None, source_name: str):
#     """Show image with OCR text"""
#     st.subheader(f"🖼️ Image Source: {source_name}")
#     st.image(image_url, use_container_width=True)
#     if ocr_text:
#         with st.expander("🔍 OCR Text"):
#             st.text_area("Extracted Text", ocr_text, height=200)


with st.sidebar:
    st.title("Medical RAG")
    if "email" not in st.session_state:
        st.session_state["email"] = None
    if "access_token" not in st.session_state:
        st.session_state["access_token"] = None
    if "chatroom_id" not in st.session_state:
        st.session_state["chatroom_id"] = None

    st.subheader("Account")
    if not st.session_state["email"]:
        mode = st.radio("Mode", ("Login", "Signup"))
        email = st.text_input("Email", key="sid_email")
        password = st.text_input("Password", type="password", key="sid_pass")
        if st.button("Submit"):
            if not email or not password:
                st.error("Enter both email and password.")
            else:
                try:
                    endpoint = "/login" if mode == "Login" else "/register"
                    r = requests.post(f"{BACKEND_URL}{endpoint}", json={"email": email, "password": password}, headers=HEADERS_BASE, timeout=20)
                    if r.ok:
                        if mode == "Login":
                            data = r.json()
                            token = data.get("access_token")
                            if token:
                                st.session_state["access_token"] = token
                                st.session_state["email"] = email
                                st.success("Logged in successfully!")
                                st.rerun()
                            else:
                                st.error("Login succeeded but token missing.")
                        else:
                            st.success("Signup successful. Please login.")
                    else:
                        try:
                            st.error(r.json().get("detail", r.text))
                        except:
                            st.error(r.text)
                except Exception as e:
                    st.error(f"Authentication failed: {e}")
    else:
        st.write(f"Logged in as **{st.session_state['email']}**")
        if st.button("Logout"):
            st.session_state["email"] = None
            st.session_state["access_token"] = None
            st.session_state["chatroom_id"] = None
            st.rerun()

    # Only for logged in users
    if st.session_state.get("access_token"):
        st.markdown("---")
        st.subheader("Chatrooms")
        try:
            r = requests.get(f"{BACKEND_URL}/chatrooms/{st.session_state['email']}", headers=authed_headers(), timeout=15)
            if r.status_code == 401:
                st.error("Session expired. Please login again.")
            elif r.ok:
                rooms = r.json().get("chatrooms", [])
                if rooms:
                    keys = [f"Chatroom {r[0]}" for r in rooms]
                    mapping = {f"Chatroom {r[0]}": r[0] for r in rooms}
                    default_index = 0
                    cur_id = st.session_state.get("chatroom_id")
                    if cur_id:
                        name = f"Chatroom {cur_id}"
                        if name in keys:
                            default_index = keys.index(name)
                    sel = st.selectbox("Select chatroom", keys, index=default_index)
                    st.session_state["chatroom_id"] = mapping[sel]
                else:
                    st.info("No chatrooms. Create one below.")
            else:
                st.error("Failed to fetch chatrooms.")
        except Exception as e:
            st.warning(f"Could not fetch chatrooms: {e}")

        if st.button("Create new chatroom"):
            try:
                r = requests.post(f"{BACKEND_URL}/new_chatroom/{st.session_state['email']}", headers=authed_headers(), timeout=15)
                if r.ok:
                    st.session_state["chatroom_id"] = r.json().get("chatroom_id")
                    st.success(f"New chatroom #{st.session_state['chatroom_id']} created")
                    st.rerun()
                elif r.status_code == 401:
                    st.error("Unauthorized. Please login again.")
                    st.session_state["email"] = None
                    st.session_state["access_token"] = None
                else:
                    st.error("Failed to create chatroom.")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.subheader("Upload Documents")
        uploaded = st.file_uploader(
            "Select files (.pdf,.docx,.csv,.png,.jpg,.mp4,.mp3)", 
            accept_multiple_files=True, 
            type=["pdf","docx","csv","png","jpg","jpeg","mp4","mp3"]
        )
        if st.button("Process Documents"):
            if not uploaded:
                st.warning("Choose files first")
            else:
                files = []
                for f in uploaded:
                    files.append(("files", (f.name, f.getvalue(), f.type)))
                try:
                    with st.spinner("Processing documents..."):
                        r = requests.post(f"{BACKEND_URL}/upload", files=files, headers=authed_headers(), timeout=300)
                    if r.status_code == 401:
                        st.error("Unauthorized. Please login again.")
                        st.session_state["email"] = None
                        st.session_state["access_token"] = None
                    elif r.ok:
                        j = r.json()
                        st.success(j.get("message", "Processed"))
                        docs = j.get("documents", [])
                        for d in docs:
                            if d.get("status") == "success":
                                st.info(f"✅ {d.get('filename')}: {d.get('chunks_count')} chunks")
                            else:
                                st.error(f"❌ {d.get('filename')}: {d.get('status')}")
                    else:
                        st.error(f"Upload failed: {r.text}")
                except Exception as e:
                    st.error(f"Upload error: {e}")

# -------------------------
# Main app area
# -------------------------
st.title("🩺 Medical RAG Chatbot")

if not st.session_state.get("access_token"):
    st.info("Please login or create an account from the sidebar.")
    st.stop()

if not st.session_state.get("chatroom_id"):
    st.info("Create or select a chatroom in the sidebar.")
    st.stop()

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load message history
try:
    r = requests.get(f"{BACKEND_URL}/messages/{st.session_state['chatroom_id']}", headers=authed_headers(), timeout=15)
    if r.status_code == 401:
        st.error("Session expired.")
    elif r.ok:
        hist = r.json().get("messages", [])
        # Convert to chat format if not already loaded
        if not st.session_state.messages and hist:
            for user_msg, ai_msg in hist:
                if user_msg:
                    st.session_state.messages.append({"role": "user", "content": user_msg})
                if ai_msg:
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    else:
        hist = []
except Exception:
    hist = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare headers for API call
    headers = authed_headers()
    headers["Content-Type"] = "application/json"

    # Stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        collected_response = ""
        sources = []

        try:
            with requests.post(f"{BACKEND_URL}/query",
                               json={
                                   "email": st.session_state["email"], 
                                   "chatroom_id": st.session_state["chatroom_id"], 
                                   "Query": prompt
                               },
                               headers=headers,
                               stream=True,
                               timeout=300) as resp:

                if resp.status_code == 401:
                    st.error("Unauthorized. Please login again.")
                    st.session_state["email"] = None
                    st.session_state["access_token"] = None
                    st.stop()

                # Parse sources from header
                sources = read_sources_header(resp)

                # Stream the response text
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        try:
                            part = chunk.decode("utf-8", errors="replace")
                        except:
                            part = str(chunk)
                        collected_response += part
                        response_placeholder.markdown(collected_response + "▌")

                # Final response (remove cursor)
                response_placeholder.markdown(collected_response.strip() or "No response generated.")

        except Exception as e:
            st.error(f"Error during chat: {e}")
            collected_response = "Sorry, there was an error processing your request."
            response_placeholder.markdown(collected_response)

    # Add assistant response to chat history
    if collected_response:
        st.session_state.messages.append({"role": "assistant", "content": collected_response})
    import re

    def extract_doc_number(answer_text: str):
        match = re.search(r"Doc\s+(\d+)", answer_text)
        if match:
            return int(match.group(1))
        return None
    # Display enhanced sources
    if sources:
            st.markdown("---")
            st.markdown("### 📚 **Source Documents**")
        
            doc_num = extract_doc_number(collected_response)
            if doc_num and 0 < doc_num <= len(sources):
                top_source = sources[doc_num - 1]   # Doc 1 → index 0
            else:
                top_source = sources[0]

            display = top_source.get("display", {}) or {}
            cs = top_source.get("complete_source", {}) or {}
            meta = top_source.get("metadata", {}) or {}
            title = cs.get("source") or meta.get("source") or f"Source {top_source}"
                
            dtype = display.get("type")
            
            # VIDEO SOURCE
            if dtype == "video_player" or (cs.get("video_url") and cs.get("video_url").lower().endswith((".mp4", ".mov", ".webm"))):
                url = cs.get("video_url") or cs.get("asset_uri")
                start = display.get("start_time") or cs.get("start")
                end = display.get("end_time") or cs.get("end")
                transcript = cs.get("transcript") or meta.get("transcript")
                display_video_segment(url, start, end, transcript, title, f"video_{top_source}")
            
            
            # PDF PAGE
            elif dtype == "pdf_page" or (cs.get("pdf_url") or (cs.get("asset_uri") and cs.get("asset_uri").lower().endswith(".pdf"))):
                pdf_url = cs.get("pdf_url") or cs.get("asset_uri")
                page_number = cs.get("page_number") or meta.get("page")
                full_text = cs.get("full_text") or meta.get("full_text")
                display_pdf_page(pdf_url, page_number, title)
            
            # CSV ROW
            elif dtype == "csv" or (cs.get("csv_url") or (cs.get("asset_uri") and cs.get("asset_uri").lower().endswith(".csv"))):
                csv_url = cs.get("csv_url") or cs.get("asset_uri")
                headers = cs.get("headers") or meta.get("csv_columns", "").split(",")
                row_number = cs.get("row_number") or meta.get("row_number")

                row_json = cs.get("complete_row") or meta.get("row_data")
                try:
                    complete_row = json.loads(row_json) if isinstance(row_json, str) else (row_json or {})
                except Exception:
                    complete_row = {}

                display_csv_row(csv_url, headers, complete_row, row_number, title)
            
            # DOCX PARAGRAPH
            elif dtype == "docx_paragraph":
                docx_url = cs.get("docx_url") or cs.get("asset_uri")
                paragraph_index = cs.get("paragraph_index") or meta.get("paragraph_index")
                full_text = cs.get("full_text") or meta.get("full_text")
                display_docx_preview(docx_url, title, paragraph_index, full_text)
            
            # IMAGE with OCR
            elif dtype == "image_viewer" or (cs.get("image_url") or (cs.get("asset_uri") and cs.get("asset_uri").lower().endswith((".png", ".jpg", ".jpeg", ".gif")))):
                img_url = cs.get("image_url") or cs.get("asset_uri")
                ocr_text = cs.get("full_text") or meta.get("full_text") or top_source.get("search_chunk")
                display_image_with_ocr(img_url, ocr_text, title)
            
            # DEFAULT/FALLBACK
            else:
                st.markdown(f"""
                <div style="border: 1px solid #ccc; border-radius: 8px; padding: 12px; background: #f8f9fa; margin: 10px 0;">
                    <h4 style="margin-top: 0;">📄 {title}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                content = cs.get("text") or top_source.get("search_chunk")
                if content:
                    st.markdown(f"""
                    <div style="background: white; padding: 12px; border-radius: 4px; border-left: 4px solid #666;">
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show asset link if available
                if cs.get("asset_uri"):
                    asset_url = normalize_url(cs.get("asset_uri"))
                    st.markdown(f"[📎 View File]({asset_url})")
            
            st.markdown("---")
