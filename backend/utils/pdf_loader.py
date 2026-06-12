import os
from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf(file_path, file_name=None):
    """
    Loads a PDF file and extracts text page-by-page.
    Returns a list of LangChain Document objects with source and page metadata.
    """
    if file_name is None:
        file_name = os.path.basename(file_path)
        
    reader = PdfReader(file_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_num + 1
                    }
                )
            )

    return documents