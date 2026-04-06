import os
import fitz  # PyMuPDF
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
load_dotenv()

# Setup PyTesseract
import pytesseract
from PIL import Image
import io

# We assume Tesseract goes here under default installation paths on Windows.
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Setup Sarvam AI
try:
    from sarvamai import SarvamAI
    api_key = os.getenv("SARVAMAI_KEY")
    if api_key and api_key != "your_key_here":
        sarvam_client = SarvamAI(api_subscription_key=api_key)
    else:
        sarvam_client = None
except ImportError:
    sarvam_client = None

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)

global_pdf_data = {
    "filename": None,
    "filepath": None,
    "toc": [],
    "pages": {}, # Cache for text per page {page_index: text}
    "total_pages": 0,
    "analysis": "No global analysis has been generated yet for this document."
}

SARVAM_MODEL = "sarvam-30b" # Replace with 'sarvam-105b' or 'sarvam-2B-chat' based on your access.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global global_pdf_data
    file_path = f"uploads/{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        doc = fitz.open(file_path)
        toc = doc.get_toc() # [level, title, page_number]
        total_pages = len(doc)
        
        global_pdf_data = {
            "filename": file.filename,
            "filepath": file_path,
            "toc": toc,
            "pages": {}, # We'll extract text lazily
            "total_pages": total_pages,
            "analysis": "No global analysis has been generated yet for this document."
        }
        
        doc.close()
        
        return {
            "status": "success",
            "filename": file.filename,
            "total_pages": total_pages,
            "toc_entries": len(toc)
        }
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Expose uploads folder for the frontend PDF reader
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def extract_page_sync(page_index: int) -> str:
    """Synchronous function to perform extraction and Tesseract OCR on a single page"""
    global global_pdf_data
    
    if page_index in global_pdf_data["pages"] and global_pdf_data["pages"][page_index]:
        return global_pdf_data["pages"][page_index]
        
    if not global_pdf_data["filepath"]:
        return ""
        
    doc = fitz.open(global_pdf_data["filepath"])
    
    if page_index < 0 or page_index >= len(doc):
        doc.close()
        return ""
        
    page = doc[page_index]
    text = page.get_text("text").strip()
    
    # Using Tesseract OCR if text isn't deeply embedded
    if len(text) < 20:
        logger.info(f"Page {page_index+1} lacks native text. Running Tesseract OCR...")
        try:
            pix = page.get_pixmap(dpi=150) # Tesseract likes slightly higher DPI
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
        except Exception as e:
            logger.error(f"Tesseract Error: {e}")
            text = "[Image could not be parsed by Tesseract]"
            
    global_pdf_data["pages"][page_index] = text
    doc.close()
    return text

async def extract_page_async(page_index: int) -> str:
    """Run the synchronous extraction in a thread pool to avoid blocking FastAPI"""
    return await asyncio.to_thread(extract_page_sync, page_index)

async def build_context(center_page: int, window: int = 5) -> str:
    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        return ""
    
    start_page = max(0, center_page - window)
    end_page = min(total - 1, center_page + window)
    
    # We await all pages in the window (Running OCR concurrently if needed)
    tasks = [extract_page_async(p) for p in range(start_page, end_page + 1)]
    results = await asyncio.gather(*tasks)
    
    context_parts = []
    for i, text in enumerate(results):
        p = start_page + i
        if text.strip():
            context_parts.append(f"--- Page {p + 1} ---\n{text.strip()}")
            
    return "\n\n".join(context_parts)

@app.post("/api/analyze_env")
async def analyze_env(request: Request):
    """
    Runs an analysis over the local environment (+/- 5 pages),
    and asks Sarvam to create a high-level summary of this section.
    """
    if not sarvam_client:
        raise HTTPException(status_code=500, detail="Sarvam AI client not configured. Check your .env API Key")
        
    data = await request.json()
    current_page = data.get("current_page", 1) - 1 # 0-indexed
    
    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
        
    extracted_text = await build_context(current_page, window=5)
    
    prompt = f"""You are a document analyzer. Convert this raw OCR text into highly structured, clean synthetic data context.
Provide a clear, high-level summary of what this specific section is about and outline its main topics.

Raw OCR Document Text:
{extracted_text}
"""
    
    try:
        response = await asyncio.to_thread(
            sarvam_client.chat.completions,
            model=SARVAM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        content = response.choices[0].message.content
        analysis_result = content.strip() if content else "[Error: API returned empty or rejected response for this context]"
        
        # Save Env conversion to distinct file
        with open("surrounding_context.txt", "w", encoding="utf-8") as f:
            f.write(analysis_result)
            
        return JSONResponse(content={"analysis": analysis_result + " (Saved to surrounding_context.txt)"})
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_global")
async def analyze_global():
    """
    Runs an analysis over the ENTIRE PDF using concurrent OCR,
    and then asks Sarvam AI for synthetic data conversion into context.txt.
    """
    if not sarvam_client:
        raise HTTPException(status_code=500, detail="Sarvam AI client not configured. Check your .env API Key")

    global global_pdf_data
    total = global_pdf_data["total_pages"]
    if total == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded")
        
    sem = asyncio.Semaphore(20)
    async def extract_with_semaphore(p):
        async with sem:
            return await extract_page_async(p)
            
    CHUNK_SIZE = 20 # Processing in 20-page chunks to avoid token limits
    
    # Initialize context.txt to clear previous data
    with open("context.txt", "w", encoding="utf-8") as f:
        f.write("")
        
    overall_analysis = ""
    
    try:
        for i in range(0, total, CHUNK_SIZE):
            chunk_end = min(i + CHUNK_SIZE, total)
            logger.info(f"Processing global analysis for pages {i+1} to {chunk_end}...")
            
            tasks = [extract_with_semaphore(p) for p in range(i, chunk_end)]
            results = await asyncio.gather(*tasks)
            
            extracted_text = "\n\n".join([r for r in results if r])
            if not extracted_text.strip():
                continue
                
            prompt = f"""Convert all of the following messy OCR text into clean, structured synthetic data context. 
Organize the overarching concepts strictly and clearly so it can be used for RAG applications.

Raw Document Text (Pages {i+1} to {chunk_end}):
{extracted_text}
"""
            
            response = await asyncio.to_thread(
                sarvam_client.chat.completions,
                model=SARVAM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
            )
            content = response.choices[0].message.content
            analysis_result = content.strip() if content else "[Error: API returned empty or rejected response for this context]"
            
            # -------------------------------------------------------------
            # VITAL REQUIREMENT: Appending structured data in chunks
            # -------------------------------------------------------------
            with open("context.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Analysis for Pages {i+1} to {chunk_end} ---\n\n")
                f.write(analysis_result)
                
            overall_analysis += f"\n\n--- Analysis for Pages {i+1} to {chunk_end} ---\n\n" + analysis_result
            
        global_pdf_data["analysis"] = overall_analysis
        return JSONResponse(content={"analysis": "Synthetic data successfully generated in chunks and appended to context.txt!"})
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("query")
    mode = data.get("mode", "analyze") # default to global analyze mode
    
    global global_pdf_data
    if global_pdf_data["total_pages"] == 0:
        raise HTTPException(status_code=400, detail="No PDF uploaded")

    if not sarvam_client:
        async def stream_error():
            yield f"data: {json.dumps({'error': 'Sarvam AI client not configured! Check your .env file or run pip install.'})}\n\n"
            yield "event: end\ndata: {}\n\n"
        return StreamingResponse(stream_error(), media_type="text/event-stream")

    # Read strictly from physical files based on requested mode
    target_file = "context.txt" if mode == "analyze" else "surrounding_context.txt"
    file_context = "No context data found! Please securely run the equivalent Analysis button first to extract and generate the data."
    
    if os.path.exists(target_file):
        with open(target_file, "r", encoding="utf-8") as f:
            file_context = f.read()
    
    system_prompt = f"""You are an advanced, helpful document assistant responding to queries.
You must strictly rely on the generated Synthetic Data context provided from {target_file}.
If the context does not contain the answer, explicitly state that you cannot find it in the current mode's context.

Extracted Synthetic Context ({target_file}):
{file_context}
"""

    user_prompt = f"Question: {query}"

    async def single_chunk_response():
        # Using synchronous fetch then yielding to simulate streaming
        try:
            response = await asyncio.to_thread(
                sarvam_client.chat.completions,
                model=SARVAM_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            content = response.choices[0].message.content
            yield f"data: {json.dumps({'content': content})}\n\n"
            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"

    return StreamingResponse(single_chunk_response(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)