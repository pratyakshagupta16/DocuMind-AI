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

# Startup configuration logging
_gemini_key = os.getenv("GEMINI_API_KEY")
_openai_key = os.getenv("OPENAI_API_KEY")
_ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
_ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print("INFO [DocuMind AI]: Initializing AI configurations...")
if _gemini_key and HAS_GEMINI:
    print("INFO [DocuMind AI]: Primary AI Provider: Google Gemini (Preferred Model: gemini-2.5-flash)")
elif _openai_key and HAS_OPENAI:
    print("INFO [DocuMind AI]: Primary AI Provider: OpenAI (Model: gpt-4o-mini)")
else:
    print(f"INFO [DocuMind AI]: Primary AI Provider: Local Ollama (Model: {_ollama_model} at {_ollama_url})")


def get_llm():
    """
    Helper function to load the local Ollama model.
    """
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return ChatOllama(
        model=ollama_model,
        base_url=ollama_url,
        temperature=0.0
    )

def invoke_llm(llm_or_prompt, prompt_or_messages=None):
    """
    Invokes the LLM with dynamic model fallback and logging.
    """
    prompt = prompt_or_messages if prompt_or_messages is not None else llm_or_prompt
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    # 1. Try Google Gemini with currently supported production models
    if gemini_key and HAS_GEMINI:
        gemini_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        for model in gemini_models:
            try:
                print(f"INFO [DocuMind AI]: Attempting model 'google/{model}' via Gemini API...")
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=gemini_key,
                    temperature=0.0
                )
                res = llm.invoke(prompt)
                print(f"INFO [DocuMind AI]: Successful response from model 'google/{model}'")
                return res
            except Exception as e:
                print(f"WARN [DocuMind AI]: Gemini model '{model}' failed: {str(e)}")
                continue

    # 2. Try OpenAI
    if openai_key and HAS_OPENAI:
        try:
            print("INFO [DocuMind AI]: Attempting model 'openai/gpt-4o-mini'...")
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_key,
                temperature=0.0
            )
            res = llm.invoke(prompt)
            print("INFO [DocuMind AI]: Successful response from model 'openai/gpt-4o-mini'")
            return res
        except Exception as e:
            print(f"WARN [DocuMind AI]: OpenAI model gpt-4o-mini failed: {str(e)}")

    # 3. Fallback to Local Ollama
    try:
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"INFO [DocuMind AI]: Attempting local Ollama model '{ollama_model}' at '{ollama_url}'...")
        
        llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_url,
            temperature=0.0
        )
        res = llm.invoke(prompt)
        print(f"INFO [DocuMind AI]: Successful response from Ollama model '{ollama_model}'")
        return res
    except Exception as e:
        raise RuntimeError(
            "⚠️ **AI Inference Failure**\n\n"
            "We were unable to generate a response because all configured AI providers failed to respond. "
            "Please check the following configurations:\n"
            "1. **Gemini API Key:** If you are using Google Gemini, make sure `GEMINI_API_KEY` is set correctly in your environment variables or Streamlit Secrets.\n"
            "2. **OpenAI API Key:** If you are using OpenAI, make sure `OPENAI_API_KEY` is set correctly.\n"
            "3. **Local Ollama:** If you are running locally without cloud API keys, ensure that Ollama is running (`ollama serve`) and the specified model is downloaded.\n\n"
            f"*Technical Error Details:* `{str(e)}`"
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