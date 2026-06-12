# DocuMind AI

**DocuMind AI** is a professional, high-performance document intelligence assistant. It allows you to chat with multiple PDF files side-by-side with an integrated PDF viewer, view transparent page citations, and generate custom study flashcards on demand. The backend runs completely locally using Ollama and FAISS embeddings.

---

## Features
- **💬 Side-by-Side PDF Chat**: Question your document database with responses grounded strictly in PDF content.
- **📖 Integrated Document Viewer**: View your uploaded documents side-by-side with the chat. Click citation references to jump directly to specific pages.
- **🔍 Transparent Citations**: Exact sources and page numbers are highlighted under answers, with raw snippets available in an expandable citation detail box.
- **📇 On-Demand Flashcards**: Generates study flashcard questions and answers with collapsible reveal views.
- **📥 Notes Export**: Download study guides, citation references, and flashcards in a formatted Markdown file.
- **⚡ Performance Caching**: PDF processing and vector store building are cached via Streamlit resources, running exactly once per distinct file-hash upload.

---

## Tech Stack
- **Frontend**: Streamlit
- **Orchestration**: LangChain (`langchain-core`, `langchain-community`)
- **Model Interface**: `langchain-ollama`
- **Embeddings**: `langchain-huggingface` (using `sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (using `faiss-cpu`)
- **PDF Parser**: `pypdf`

---

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### Local Setup
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd DocuMind
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On MacOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start local Ollama & Pull Model**:
   Make sure the Ollama application is running, and pull your model of choice:
   ```bash
   ollama pull llama3
   ```

5. **Set Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   OLLAMA_MODEL=llama3
   OLLAMA_BASE_URL=http://localhost:11434
   ```

---

## Running the Application
Run the Streamlit application:
```bash
streamlit run frontend/app.py
```

---

## Screenshots Placeholder
*(Add your app interface screenshots here)*

---

## Deployment Strategy & Limitations

### The Ollama Deployment Constraint
Ollama runs LLMs locally on system hardware. Free-tier cloud instances (like Render Free, Railway, or Streamlit Community Cloud) do not include GPUs and limit RAM to 512MB–2GB. Running a model like `llama3` requires at least 4.6GB of RAM and significant CPU/GPU compute, meaning **Ollama cannot run inside a standard free cloud container**.

### Recommended Hosting Options:

#### 1. Self-Hosted VPS with GPU (Recommended)
Deploy the Streamlit frontend container on a VM (AWS EC2, DigitalOcean, RunPod, or Vast.ai) with GPU access, and run Ollama alongside it.

#### 2. Hybrid Deployment (Streamlit Cloud + Cloud Ollama)
Host the Streamlit app on Streamlit Community Cloud (free) or Render, and configure the `OLLAMA_BASE_URL` environment variable to point to a secure remote Ollama instance running on a private VPS or GPU cloud (e.g. RunPod endpoint).

#### 3. Shift to Cloud APIs (Alternative)
If you want serverless cloud hosting without a private VPS, modify the LLM wrapper in `backend/utils/generator.py` to use a cloud API provider, such as:
- **Gemini API**: Replace `ChatOllama` with `ChatGoogleGenerativeAI` from `langchain_google_genai`.
- **OpenAI API**: Replace `ChatOllama` with `ChatOpenAI` from `langchain_openai`.