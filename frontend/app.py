import sys
import os
import hashlib
import time
import base64
import html

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'backend')
    )
)

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from utils.pdf_loader import load_pdf
from utils.embedder import split_documents
from utils.vector_store import (
    get_embeddings,
    create_vector_store,
    load_local_vector_store,
    save_local_vector_store
)
from utils.generator import (
    generate_answer,
    generate_flashcards
)

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="DocuMind AI - Study Companion",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# ---------------- INITIALIZE SESSION STATE ---------------- #
if "chats" not in st.session_state:
    st.session_state.chats = {
        "chat_1": {
            "name": "Default Chat",
            "messages": []
        }
    }
    st.session_state.active_chat_id = "chat_1"

if "processed_files_hash" not in st.session_state:
    st.session_state.processed_files_hash = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "revealed_cards" not in st.session_state:
    st.session_state.revealed_cards = {}

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-color: #0b0f19;
    --sidebar-bg: #0f172a;
    --text-color: #f8fafc;
    --chat-bubble-user: #1e293b;
    --chat-bubble-assistant: #0f172a;
    --border-color: #1e293b;
    --citation-bg: #0b0f19;
    --text-muted: #94a3b8;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stDeployButton"] {
    display: none !important;
}

/* Base fonts and spacing */
.stApp, [data-testid="stAppViewContainer"], .main, body {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    font-family: 'Outfit', 'Inter', sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stButton button {
    background-color: var(--bg-color);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

section[data-testid="stSidebar"] .stButton button:hover {
    border-color: #6366f1;
    color: #6366f1;
    background-color: var(--sidebar-bg);
}

/* Title gradient */
.header-container {
    text-align: center;
    padding: 15px 10px;
    margin-bottom: 15px;
}

.gradient-title {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 5px;
}

.header-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
}

/* Chat bubble styling */
.stChatMessage {
    padding: 1.25rem 1.5rem !important;
    border-radius: 12px !important;
    margin-bottom: 1.25rem !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
}

.stChatMessage[data-testid="stChatMessageUser"] {
    background-color: var(--chat-bubble-user) !important;
}

.stChatMessage[data-testid="stChatMessageAssistant"] {
    background-color: var(--chat-bubble-assistant) !important;
}

.stChatMessage p, .stChatMessage span, .stChatMessage li, .stChatMessage div {
    color: var(--text-color) !important;
}

/* Citations */
.source-citation-box {
    background-color: var(--citation-bg);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color);
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.5;
    margin-bottom: 12px;
}

.source-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    background-color: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 6px;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

/* Flashcard Premium Styling */
.flashcard-deck-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.active-chat-badge {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    padding: 8px 12px;
    border-radius: 8px;
    font-weight: 600;
    color: white;
    margin-bottom: 6px;
    border: 1px solid rgba(124, 58, 237, 0.4);
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPER HASHING FUNCTION ---------------- #
def get_files_hash(uploaded_files):
    if not uploaded_files:
        return ""
    hasher = hashlib.md5()
    for f in uploaded_files:
        hasher.update(f.name.encode('utf-8'))
        hasher.update(str(f.size).encode('utf-8'))
        hasher.update(f.getvalue()[:1024*1024])
    return hasher.hexdigest()

# ---------------- CACHED PDF LOADER & INDEX BUILDER ---------------- #
@st.cache_resource(show_spinner=False)
def get_vector_db_for_files(files_hash_str, uploaded_files):
    embeddings = get_embeddings()
    db_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vector_store", files_hash_str)
    )
    
    if os.path.exists(db_path):
        return load_local_vector_store(db_path, embeddings)
        
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            docs = load_pdf(temp_path, uploaded_file.name)
            all_docs.extend(docs)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    chunks = split_documents(all_docs)
    vector_db = create_vector_store(chunks, embeddings)
    save_local_vector_store(vector_db, db_path)
    return vector_db

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0;'>
        <h2 style='margin:0; font-weight:800; color:white; font-size:1.6rem;'>📄 DocuMind AI</h2>
        <span style='color:#818cf8; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Document Assistant</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload PDFs
    st.markdown("### 📥 Upload Source PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    pdf_files = []
    if uploaded_files:
        pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]

    # Active Documents List
    if pdf_files:
        st.markdown("### 📄 Active Documents")
        for f in pdf_files:
            st.markdown(f"<div style='font-size:0.82rem; color:var(--text-muted); padding: 2px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>• {f.name}</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Chat History / Sessions
    st.markdown("### 💬 Chat Sessions")
    
    # New Chat Session Button
    if st.button("➕ New Chat Session", use_container_width=True):
        new_id = f"chat_{int(time.time())}"
        st.session_state.chats[new_id] = {
            "name": f"New Chat {len(st.session_state.chats) + 1}",
            "messages": []
        }
        st.session_state.active_chat_id = new_id
        st.rerun()
        
    # List Chat Sessions
    chat_ids = list(st.session_state.chats.keys())
    for cid in chat_ids:
        cname = st.session_state.chats[cid]["name"]
        if cid == st.session_state.active_chat_id:
            st.markdown(f"<div class='active-chat-badge'>💬 {cname}</div>", unsafe_allow_html=True)
        else:
            if st.button(f"💬 {cname}", key=f"select_{cid}", use_container_width=True):
                st.session_state.active_chat_id = cid
                st.rerun()
                
    st.markdown("---")
    
    # Rename and Delete Active Chat
    active_chat = st.session_state.chats[st.session_state.active_chat_id]
    st.markdown("##### Manage Active Chat")
    new_name = st.text_input("Rename Chat", value=active_chat["name"], label_visibility="collapsed")
    if new_name != active_chat["name"] and new_name.strip() != "":
        active_chat["name"] = new_name.strip()
        st.rerun()
        
    if len(st.session_state.chats) > 1:
        if st.button("🗑️ Delete Active Chat", use_container_width=True, type="secondary"):
            del st.session_state.chats[st.session_state.active_chat_id]
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
            st.rerun()

# ---------------- VECTOR STORE PROCESSING & CACHING ---------------- #
if pdf_files:
    files_hash_str = get_files_hash(pdf_files)
    
    if st.session_state.processed_files_hash != files_hash_str:
        with st.status("⚡ Preparing document workspace...", expanded=True) as status_box:
            status_box.write("Processing documents and loading local vector index...")
            vector_db = get_vector_db_for_files(files_hash_str, pdf_files)
            st.session_state.vector_db = vector_db
            
            # Reset active chats since files changed
            st.session_state.chats = {
                "chat_1": {
                    "name": "Default Chat",
                    "messages": []
                }
            }
            st.session_state.active_chat_id = "chat_1"
            st.session_state.revealed_cards = {}
            st.session_state.processed_files_hash = files_hash_str
            status_box.update(label="⚡ Workspace ready!", state="complete", expanded=False)
else:
    st.session_state.vector_db = None
    st.session_state.processed_files_hash = None

# ---------------- MAIN CONTENT AREA & STUDY NOTES EXPORT ---------------- #
st.markdown("""
<div class='header-container'>
    <h1 class='gradient-title'>DocuMind AI</h1>
    <p class='header-subtitle'>Chat with PDFs. Learn Faster.</p>
</div>
""", unsafe_allow_html=True)

# Helper function to generate notes Markdown string
def generate_notes_markdown(messages):
    md = "# DocuMind Study Notes & Summary\n\n"
    for idx, msg in enumerate(messages):
        if msg["role"] == "user":
            md += f"### ❓ Query: {msg['content']}\n\n"
        else:
            md += f"**Answer:**\n{msg['content']}\n\n"
            if "sources" in msg and msg["sources"]:
                unique_srcs = {}
                for src in msg["sources"]:
                    unique_srcs[f"{src['source']} (Page {src['page']})"] = True
                md += f"*Sources:* {', '.join(unique_srcs.keys())}\n\n"
            
            if "flashcards" in msg and msg["flashcards"]:
                md += "**Generated Study Flashcards:**\n"
                for fc in msg["flashcards"]:
                    md += f"- **Q:** {fc['question']}\n"
                    md += f"  **A:** {fc['answer']}\n"
                md += "\n"
            md += "---\n\n"
    return md

# Layout configuration
if pdf_files:
    col_chat, col_preview = st.columns([3, 2])
else:
    col_chat = st.container()
    col_preview = None

# Render Chat Column
with col_chat:
    active_chat = st.session_state.chats[st.session_state.active_chat_id]
    messages = active_chat["messages"]
    
    if pdf_files:
        st.caption(f"📁 Active Source: {len(pdf_files)} PDF(s) loaded")
        
        # Download study notes button
        if messages:
            notes_md = generate_notes_markdown(messages)
            st.download_button(
                label="📥 Export Notes (Markdown)",
                data=notes_md,
                file_name="documind_study_notes.md",
                mime="text/markdown",
                use_container_width=True
            )
            
    # Onboarding Empty State
    if not messages:
        st.markdown("<h4 style='text-align: center; color: var(--text-muted); margin-top: 30px; margin-bottom: 20px;'>💡 Ask a question or try one of these suggestions:</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        suggestions = [
            "Summarize this document.",
            "What are the key takeaways?",
            "Explain this topic in simple terms.",
            "Generate study flashcards.",
            "What are the important concepts?"
        ]
        
        for sug_idx, sug in enumerate(suggestions):
            with col1 if sug_idx % 2 == 0 else col2:
                if st.button(f"✨ {sug}", key=f"sug_{sug_idx}", use_container_width=True):
                    messages.append({
                        "role": "user",
                        "content": sug
                    })
                    st.rerun()
    else:
        # Render chat history
        for idx, message in enumerate(messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    
                    # Transparent source citations
                    if "sources" in message and message["sources"]:
                        unique_sources = {}
                        for src in message["sources"]:
                            unique_sources[f"{src['source']} (Page {src['page']})"] = (src['source'], src['page'])
                        
                        sources_str = ", ".join(unique_sources.keys())
                        st.markdown(f"<div style='font-size:0.85rem; color:var(--text-muted); margin-top: -5px; margin-bottom: 8px;'>🔍 **Sources:** {sources_str}</div>", unsafe_allow_html=True)
                        
                        # Preview controls inline with citation
                        cols_c = st.columns(len(unique_sources) + 1)
                        for c_idx, (src_lbl, (src_name, src_page)) in enumerate(unique_sources.items()):
                            with cols_c[c_idx]:
                                if st.button(f"📖 Preview {src_lbl}", key=f"preview_btn_{idx}_{c_idx}"):
                                    st.session_state.preview_pdf_name = src_name
                                    st.session_state.preview_page_num = src_page
                                    st.rerun()
                                    
                        # Detailed expander
                        with st.expander("📄 View Citation Details", expanded=False):
                            for source_info in message["sources"]:
                                src_name = source_info.get("source", "Unknown PDF")
                                src_page = source_info.get("page", "Unknown")
                                st.markdown(f"<span class='source-badge'>{src_name} — Page {src_page}</span>", unsafe_allow_html=True)
                                st.markdown(f"<div class='source-citation-box'>{source_info.get('text')}</div>", unsafe_allow_html=True)
                                
                    # Copy to Clipboard Button
                    escaped_text = html.escape(message["content"]).replace("'", "\\'").replace("\n", "\\n")
                    copy_html = f"""
                    <div style="text-align: right; margin-top: 5px;">
                        <button id="copy-btn-{idx}" style="background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; cursor: pointer;" onclick="copyToClipboard{idx}()">📋 Copy Answer</button>
                    </div>
                    <script>
                    function copyToClipboard{idx}() {{
                        const el = document.createElement('textarea');
                        el.value = `{escaped_text}`;
                        document.body.appendChild(el);
                        el.select();
                        document.execCommand('copy');
                        document.body.removeChild(el);
                        const btn = document.getElementById('copy-btn-{idx}');
                        btn.innerText = '✅ Copied!';
                        setTimeout(() => btn.innerText = '📋 Copy Answer', 2000);
                    }}
                    </script>
                    """
                    st.components.v1.html(copy_html, height=45)
                    
                    # Study Flashcards Redesigned
                    if "flashcards" in message and message["flashcards"]:
                        st.markdown("<div class='flashcard-deck-title'>📇 Study Flashcards</div>", unsafe_allow_html=True)
                        for card_idx, card in enumerate(message["flashcards"]):
                            card_key = f"{st.session_state.active_chat_id}_{idx}_{card_idx}"
                            is_revealed = st.session_state.revealed_cards.get(card_key, False)
                            
                            st.markdown(f"""
                            <div style='background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-top: 10px; margin-bottom: 5px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);'>
                                <div style='font-size: 0.8rem; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>Flashcard {card_idx + 1}</div>
                                <div style='font-size: 1.05rem; font-weight: 600; color: #f8fafc; line-height: 1.4;'>❓ {card['question']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Expand / Collapse control
                            if not is_revealed:
                                if st.button("👁️ Show Answer", key=f"btn_rev_{card_key}"):
                                    st.session_state.revealed_cards[card_key] = True
                                    st.rerun()
                            else:
                                st.markdown(f"""
                                <div style='background-color: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; border-radius: 6px; padding: 12px 16px; color: #34d399; font-size: 0.95rem; margin-bottom: 10px; line-height: 1.4;'>
                                    🔑 <strong>Answer:</strong> {card['answer']}
                                </div>
                                """, unsafe_allow_html=True)
                                if st.button("🙈 Hide Answer", key=f"btn_hid_{card_key}"):
                                    st.session_state.revealed_cards[card_key] = False
                                    st.rerun()
                    elif "sources" in message:
                        if st.button("✨ Generate Study Flashcards", key=f"btn_gen_fc_{idx}"):
                            with st.spinner("✨ Creating custom study cards..."):
                                from langchain_core.documents import Document
                                context_docs = []
                                for src in message.get("sources", []):
                                    context_docs.append(Document(page_content=src["text"], metadata={"source": src["source"], "page": src["page"]}))
                                
                                user_query = ""
                                if idx > 0 and messages[idx-1]["role"] == "user":
                                    user_query = messages[idx-1]["content"]
                                    
                                cards = generate_flashcards(user_query, message["content"], context_docs)
                                message["flashcards"] = cards
                                st.rerun()
                                
                    # Regenerate Button (Only for the last assistant response)
                    if idx == len(messages) - 1:
                        if st.button("🔄 Regenerate Response", key=f"btn_regen_{idx}"):
                            user_query = ""
                            for m in reversed(messages[:-1]):
                                if m["role"] == "user":
                                    user_query = m["content"]
                                    break
                            
                            if user_query:
                                with st.spinner("🤖 Regenerating response..."):
                                    docs = []
                                    if st.session_state.vector_db:
                                        docs = st.session_state.vector_db.similarity_search(user_query, k=4)
                                        
                                    new_ans = generate_answer(user_query, docs)
                                    
                                    new_sources = []
                                    for doc in docs:
                                        new_sources.append({
                                            "source": doc.metadata.get("source", "Unknown PDF"),
                                            "page": doc.metadata.get("page", "Unknown"),
                                            "text": doc.page_content
                                        })
                                    
                                    message["content"] = new_ans
                                    message["sources"] = new_sources
                                    if "flashcards" in message:
                                        del message["flashcards"]
                                    st.rerun()

    # Chat input
    query = st.chat_input("Ask a question about your PDFs...")
    
    if query:
        messages.append({
            "role": "user",
            "content": query
        })
        st.rerun()

    # Generate response if last message is from user
    if messages and messages[-1]["role"] == "user":
        last_msg = messages[-1]
        user_query = last_msg["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing query..."):
                docs = []
                if st.session_state.vector_db:
                    docs = st.session_state.vector_db.similarity_search(user_query, k=4)
                    
                ans_text = generate_answer(user_query, docs)
                
                sources_list = []
                for doc in docs:
                    sources_list.append({
                        "source": doc.metadata.get("source", "Unknown PDF"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "text": doc.page_content
                    })
                    
        messages.append({
            "role": "assistant",
            "content": ans_text,
            "sources": sources_list
        })
        st.rerun()

# Render PDF Preview Column
if col_preview is not None and pdf_files:
    with col_preview:
        st.markdown("### 📖 Document Viewer")
        
        pdf_names = [f.name for f in pdf_files]
        
        if "preview_pdf_name" not in st.session_state or st.session_state.preview_pdf_name not in pdf_names:
            st.session_state.preview_pdf_name = pdf_names[0]
        if "preview_page_num" not in st.session_state:
            st.session_state.preview_page_num = 1
            
        selected_pdf_name = st.selectbox(
            "Select Document",
            options=pdf_names,
            index=pdf_names.index(st.session_state.preview_pdf_name),
            key="preview_pdf_selector"
        )
        st.session_state.preview_pdf_name = selected_pdf_name
        
        uploaded_file = next(f for f in pdf_files if f.name == selected_pdf_name)
        
        # Read total pages cleanly
        from pypdf import PdfReader
        try:
            reader = PdfReader(uploaded_file)
            total_pages = len(reader.pages)
        except:
            total_pages = 100
            
        page_num = st.number_input(
            "Page Number",
            min_value=1,
            max_value=total_pages,
            value=int(st.session_state.preview_page_num),
            step=1,
            key="preview_page_selector"
        )
        st.session_state.preview_page_num = page_num
        
        # Render using streamlit-pdf-viewer securely
        pdf_viewer(
            input=uploaded_file.getvalue(),
            pages_to_render=[int(st.session_state.preview_page_num)],
            width="100%"
        )