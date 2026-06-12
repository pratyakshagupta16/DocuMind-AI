import os
import re
import json
import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

load_dotenv()

def get_llm():
    """
    Helper function to load the LLM. 
    Selects Gemini or OpenAI if keys are present (useful for cloud deployments), 
    otherwise falls back to local Ollama.
    """
    # 1. Check for Gemini (recommended for free-tier Streamlit Cloud)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and HAS_GEMINI:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=gemini_key,
            temperature=0.0
        )
        
    # 2. Check for OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and HAS_OPENAI:
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.0
        )
        
    # 3. Fallback to Local Ollama
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return ChatOllama(
        model=ollama_model,
        base_url=ollama_url,
        temperature=0.0
    )

def invoke_llm(llm, prompt_or_messages):
    """
    Invokes the LLM, returning results directly or raising a friendly error if it fails.
    """
    try:
        return llm.invoke(prompt_or_messages)
    except Exception as e:
        raise RuntimeError(
            f"⚠️ **AI Model Connection/Inference Failure**\n\n"
            f"Failed to communicate with your AI model provider.\n\n"
            f"Error details: `{str(e)}`"
        )


def clean_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Extract JSON object or array
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def generate_answer(query, docs, base64_images=None):
    """
    Generates an answer based on the query, context docs, and optional base64 images.
    Returns a plain string containing the markdown-formatted answer.
    """
    llm = get_llm()
    
    if llm is None:
        return "Error: Local Ollama model is not configured or available."

    # Format PDF context
    context = ""
    if docs:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", "Unknown")
            context_parts.append(f"--- Source: {source} (Page {page}) ---\n{doc.page_content}")
        context = "\n\n".join(context_parts)

    prompt = f"""You are a professional, highly accurate AI document assistant.
Your goal is to answer the user's question based strictly on the provided context from their uploaded documents and any uploaded images.

RULES:
1. Provide a direct response first, then structure the explanation according to the question type:
   - If the user asks "What is...", start with a clear, direct definition, followed by a brief, concise explanation.
   - If the user asks "Explain...", provide a detailed explanation using clean bullet points.
   - If the user asks "Summarize...", output a structured summary with 3-4 key takeaways.
   - If the user asks "List...", output a clean bullet-point list.
   - If the user asks "Compare...", output a clean Markdown comparison table.
2. Format all responses using professional, clean Markdown.
3. Do not copy large sections from the documents verbatim. Summarize and explain clearly in your own words.
4. If the answer is not present in the provided context, respond exactly: "I could not find this information in the uploaded documents." Do not try to extrapolate or use outside knowledge.
5. Do not explicitly refer to "the context" or "the provided documents" in your answer. Keep the response clean.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    content = [{"type": "text", "text": prompt}]
    if base64_images:
        for b64 in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    try:
        response = invoke_llm(llm, [{"role": "user", "content": content}])
        return response.content.strip()
    except Exception as e:
        return str(e)


def generate_flashcards(query, answer_text, docs):
    """
    Generates a set of 3-5 study flashcards based on the query, the generated answer, and context docs.
    Returns a list of dictionaries with "question" and "answer" keys.
    """
    llm = get_llm()
    if llm is None:
        return []

    context = ""
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs[:4]])

    prompt = f"""You are an educational assistant. Generate 3-5 high-quality question-answer study flashcards based on the user's query, the provided document context, and the generated answer.

RULES:
1. Every flashcard must consist of a concise question testing memory of a key concept, and a short 1-sentence answer.
2. Output strictly a JSON array of objects, where each object has "question" and "answer" keys.
3. Do not include markdown code blocks like ```json or any conversational prefix/suffix. Just return raw JSON.

Example output:
[
  {{"question": "What is the primary key concept?", "answer": "The primary key concept is..."}}
]

CONTEXT:
{context}

USER QUERY:
{query}

GENERATED ANSWER:
{answer_text}

JSON RESPONSE:"""

    try:
        response = invoke_llm(llm, prompt)
        text = response.content.strip()
        clean_text = clean_json_response(text)
        return json.loads(clean_text)
    except Exception as e:
        return [
            {
                "question": "Could not parse study cards.",
                "answer": f"Parsing failed: {str(e)}"
            }
        ]