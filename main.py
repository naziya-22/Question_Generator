from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Union
import json
import uuid
import time
from datetime import datetime
import fitz  # PyMuPDF
import os
import shutil
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# New model manager class for Google Generative AI
class ModelManager:
    """
    Class to manage Google Generative AI models used by the application
    """
    @staticmethod
    def get_model(model_name: str = "gemini-1.5-pro-002", temperature: float = 0.2):
        """Create and return a Google Generative AI model instance with the specified parameters"""
        # Set the model to output structured data for MCQ generation
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            max_retries=3
        )

    @staticmethod
    def get_chat_model(model_name: str = "gemini-1.5-pro-002", temperature: float = 0.7):
        """Create and return a Google Generative AI model instance configured for chat"""
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            max_retries=3
        )

app = FastAPI(
    title="MCQ Question Generator API",
    description="API that generates multiple choice questions using Google's Generative AI, with chat and PDF features",
    version="1.1.0"
)

# Create in-memory cache for requests
notes_question_cache = {}
notes_question_cache = {}

class SubtopicSummary(BaseModel):
    Sub_topic: str = Field(
        description="Subtitle or heading for a section within the PDF notes."
    )
    summary: str = Field(
        description="Detailed notes or explanation under this subtitle."
    )

class PDFNotes(BaseModel):

    Topic: List[SubtopicSummary] = Field(
        description="List of notes organized by subtitles for the PDF."
    )

class MCQQuestion(BaseModel):
    questionText: str = Field(
        description="The full text of the multiple choice question"
    )
    option1: str = Field(
        description="First answer option (may or may not be correct)"
    )
    option2: str = Field(
        description="Second answer option (may or may not be correct)"
    )
    option3: str = Field(
        description="Third answer option (may or may not be correct)"
    )
    option4: str = Field(
        description="Fourth answer option (may or may not be correct)"
    )
    correctOption: Literal["option1", "option2", "option3", "option4"] = Field(
        description="The single correct option (option1, option2, option3, or option4)."
    )

class QuestionBank(BaseModel):
    mcqQuestions: List[MCQQuestion] = Field(
        description="List of multiple choice questions"
    )

class GenerationRequest(BaseModel):
    topic: str = Field(
        default="Plant Life",
        description="Topic for the multiple choice questions"
    )
    count: int = Field(
        default=5, 
        ge=1, 
        le=20,
        description="Number of questions to generate (1-20)"
    )
    grade_level: str = Field(
        default="elementary",
        description="Grade level (e.g., elementary, middle, high school)"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for model generation (0.0-1.0)"
    )
    model: str = Field(
        default="gemini-1.5-pro-002",
        description="Google Generative AI model to use"
    )

class ChatRequest(BaseModel):
    message: str = Field(
        description="Message to send to the chat model"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID to continue an existing conversation"
    )
    model: str = Field(
        default="gemini-1.5-pro-002",
        description="Google Generative AI model to use"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for model generation (0.0-1.0)"
    )

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    created_at: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Union[QuestionBank, PDFNotes] = None
    error: Optional[str] = None

# In-memory conversation history cache
chat_history = {}

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file using PyMuPDF (fitz)"""
    text = ""
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text() + "\n\n"
        
        # Close the document
        doc.close()
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def generate_questions(topic: str, count: int, grade_level: str, temperature: float, task_id: str, model: str="gemini-1.5-pro-002"):
    """Background task to generate questions"""
    try:
        # Get model using the ModelManager instead of direct instantiation
        llm = ModelManager.get_model(model_name=model, temperature=temperature)
        
        # Create the prompt for the model
        prompt = f"""
    Generate {count} multiple choice questions about {topic} for {grade_level} school students.

    REQUIREMENTS:
    1. EVERY question MUST have exactly FOUR options labeled 'option1', 'option2', 'option3', and 'option4'.
    2. EVERY question MUST have exactly ONE correct option, indicated in the 'correctOption' field with the corresponding label ('option1', 'option2', 'option3', or 'option4').
    3. The questions should be age-appropriate for {grade_level} school students.
    4. Format your response as JSON with the following structure:

    {{
      "mcqQuestions": [
        {{
          "questionText": "What is photosynthesis?",
          "option1": "A process where plants release oxygen",
          "option2": "A process where plants absorb water",
          "option3": "A process where plants make their own food using sunlight",
          "option4": "A process where plants grow taller",
          "correctOption": "option3"
        }},
        {{
          "questionText": "Another question here?",
          "option1": "Option 1",
          "option2": "Option 2",
          "option3": "Option 3",
          "option4": "Option 4",
          "correctOption": "option1"
        }}
      ]
    }}

    IMPORTANT: Include the 'correctOption' field for EVERY question with the single correct option label.
    Return ONLY the JSON without any additional text or comments.
    """
        
        # Get response
        response = llm.invoke(prompt)
        
        # Parse the response
        try:
            # Extract JSON content from the response
            # Sometimes LLMs include additional text or markdown before/after the JSON
            json_content = extract_json_from_response(response.content)
            
            # Parse the JSON
            data = json.loads(json_content)
            question_bank = QuestionBank(**data)
            
            # Update cache with success
            notes_question_cache[task_id]["status"] = "completed"
            notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
            notes_question_cache[task_id]["result"] = question_bank
            
        except Exception as e:
            # Update cache with error
            notes_question_cache[task_id]["status"] = "failed"
            notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
            notes_question_cache[task_id]["error"] = f"Error parsing response: {str(e)}"
            
    except GoogleAPIError as e:
        # Handle Google API-specific errors
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Google API Error: {str(e)}"
    except Exception as e:
        # Update cache with error
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Error during generation: {str(e)}"

def extract_json_from_response(response):
    """Extract JSON content from the model response"""
    try:
        # First, try to see if the entire response is valid JSON
        json.loads(response)
        return response
    except json.JSONDecodeError:
        # If not, look for JSON-like content between curly braces
        import re
        json_pattern = r'({[\s\S]*})'
        matches = re.findall(json_pattern, response)
        
        if matches:
            for potential_json in matches:
                try:
                    # Try to parse it as JSON
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    continue
                    
        # If we're here, none of the matches were valid JSON
        raise ValueError("Could not extract valid JSON from the response")

def generate_questions_from_pdf(pdf_text: str, count: int, grade_level: str, temperature: float, task_id: str, model: str="gemini-1.5-pro-002"):
    """Background task to generate questions from PDF content"""
    try:
        # Get model using the ModelManager instead of direct instantiation
        llm = ModelManager.get_model(model_name=model, temperature=temperature)
        
        # Create the prompt for the model - limit text to prevent token overflow
        text_limit = min(len(pdf_text), 9000)  # Limit text to 9000 chars
        
        prompt = f"""
    Based on the following text content from a PDF, generate {count} multiple choice questions
    appropriate for {grade_level} school students.

    PDF CONTENT:
    {pdf_text[:text_limit]}

    REQUIREMENTS:
    1. EVERY question MUST have exactly FOUR options labeled 'option1', 'option2', 'option3', and 'option4'.
    2. EVERY question MUST have exactly ONE correct option, indicated in the 'correctOption' field with the corresponding label ('option1', 'option2', 'option3', or 'option4').
    3. The questions should be focused on the main concepts from the PDF content.
    4. There should be exactly {count} questions.
    5. Format your response as JSON with the following structure:

    {{
      "mcqQuestions": [
        {{
          "questionText": "Question text from the PDF content?",
          "option1": "Option 1",
          "option2": "Option 2",
          "option3": "Option 3",
          "option4": "Option 4",
          "correctOption": "option3"
        }},
        {{
          "questionText": "Another question from the PDF content?",
          "option1": "Option 1",
          "option2": "Option 2",
          "option3": "Option 3",
          "option4": "Option 4",
          "correctOption": "option1"
        }}
      ]
    }}

    IMPORTANT: Include the 'correctOption' field for EVERY question with the single correct option label.
    Return ONLY the JSON without any additional text or comments.
    """
        
        # Get response
        response = llm.invoke(prompt)
        
        # Parse the response
        try:
            # Extract JSON content from the response
            json_content = extract_json_from_response(response.content)
            
            # Parse the JSON
            data = json.loads(json_content)
            question_bank = QuestionBank(**data)
            
            # Update cache with success
            notes_question_cache[task_id]["status"] = "completed"
            notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
            notes_question_cache[task_id]["result"] = question_bank
            
        except Exception as e:
            # Update cache with error
            notes_question_cache[task_id]["status"] = "failed"
            notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
            notes_question_cache[task_id]["error"] = f"Error parsing response: {str(e)}"
            
    except GoogleAPIError as e:
        # Handle Google API-specific errors
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Google API Error: {str(e)}"
    except Exception as e:
        # Update cache with error
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Error during generation: {str(e)}"

def generate_notes_from_pdf(pdf_text: str, notes_type: Literal["Quick Notes", "Detailed Notes"], grade_level: str, temperature: float, task_id: str, model: str="gemini-1.5-pro-002"):
    """Background task to generate questions from PDF content"""
    try:
        # Get model using the ModelManager instead of direct instantiation
        llm = ModelManager.get_model(model_name=model, temperature=temperature)
        
        # Create the prompt for the model - limit text to prevent token overflow
        text_limit = min(len(pdf_text), 10000)  # Limit text to 9000 chars
        
        prompt = f"""
        Based on the following text content from a PDF, generate structured notes in the form of subtitles and detailed explanations.

        Notes should be appropriate for {grade_level} school students and match the {notes_type} style:

        1. Quick Notes: Short summaries under each subtitle.
        2. Detailed Notes: Thorough explanations under each subtitle.

        PDF CONTENT:
        {pdf_text[:text_limit]}
        
        RREQUIREMENTS:

        1. Organize the notes into multiple sections, each with a clear and concise subtitle.
        2. Under each subtitle, write a content field that matches the {notes_type} (quick or detailed).
        3. Cover the main ideas and important concepts from the PDF text.
        4. Keep the notes easy to understand and suitable for {grade_level} school students.

        Output Format:
        Format your response as JSON with the following structure:

        {{
        "Topic": [
            {{
            "Sub_topic": "<Subtitle for a topic/section>",
            "summary": "<Explanation or summary based on the notes_type>"
            }},
            ...
        ]
        }}

        IMPORTANT: Return ONLY the JSON without any additional text or comments.
        """
        
        # Get response
        response = llm.invoke(prompt)
        
        try:
            # Extract JSON content from the model response
            json_content = extract_json_from_response(response.content)
            
            # Parse the JSON
            data = json.loads(json_content)
            notes_bank = PDFNotes(**data)  # Assuming you have NotesBank class (similar to QuestionBank)
            
            # Update cache with success
            notes_question_cache[task_id]["status"] = "completed"
            notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
            notes_question_cache[task_id]["result"] = notes_bank
            
        except Exception as e:
            # Update cache with parsing error
            notes_question_cache[task_id]["status"] = "failed"
            notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
            notes_question_cache[task_id]["error"] = f"Error parsing response: {str(e)}"
            
    except GoogleAPIError as e:
        # Handle Google API-specific errors
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Google API Error: {str(e)}"
        
    except Exception as e:
        # Update cache with other errors
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Error during generation: {str(e)}"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the LLM model"""
    
    # Generate new conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Retrieve conversation history or initialize new one
    if conversation_id not in chat_history:
        chat_history[conversation_id] = []
    
    # Get chat model using the ModelManager instead of direct instantiation
    llm = ModelManager.get_chat_model(model_name=request.model, temperature=request.temperature)
    
    # Build context from conversation history
    context = ""
    if chat_history[conversation_id]:
        for entry in chat_history[conversation_id][-5:]:  # Use last 5 messages for context
            context += f"Human: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
    
    # Create prompt with context
    if context:
        prompt = f"{context}Human: {request.message}\nAssistant:"
    else:
        prompt = f"Human: {request.message}\nAssistant:"
    
    try:
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Store in history
        chat_history[conversation_id].append({
            "user": request.message,
            "assistant": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "conversation_id": conversation_id,
            "response": response.content,
            "created_at": datetime.now().isoformat()
        }
    except GoogleAPIError as e:
        raise HTTPException(status_code=500, detail=f"Google API Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/generate", response_model=TaskResponse)
async def create_questions(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate multiple choice questions asynchronously"""
    # Create a task ID
    task_id = str(uuid.uuid4())
    
    # Add task to cache
    notes_question_cache[task_id] = {
        "task_id": task_id,
        "status": "in_progress",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None
    }
    
    # Start the background task
    background_tasks.add_task(
        generate_questions,
        request.topic,
        request.count,
        request.grade_level,
        request.temperature,
        task_id,
        request.model
    )
    
    return {
        "task_id": task_id,
        "status": "in_progress",
        "message": f"Generation of {request.count} questions about {request.topic} started"
    }

@app.post("/generate-from-pdf", response_model=TaskResponse)
async def create_questions_from_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    count: int = Form(5),
    grade_level: str = Form("elementary"),
    temperature: float = Form(0.2),
    model: str = Form("gemini-1.5-pro-002")
):
    """Generate multiple choice questions from a PDF file"""
    # Create a task ID
    task_id = str(uuid.uuid4())
    
    # Add task to cache
    notes_question_cache[task_id] = {
        "task_id": task_id,
        "status": "in_progress",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
        "pdf_filename": pdf_file.filename
    }
    
    try:
        # Create a temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save the uploaded PDF temporarily
        pdf_path = f"temp/{task_id}_{pdf_file.filename}"
        with open(pdf_path, "wb") as file:
            shutil.copyfileobj(pdf_file.file, file)
        
        # Extract text from PDF using PyMuPDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Delete the temporary PDF file
        os.remove(pdf_path)
        
        if not pdf_text.strip():
            raise Exception("Could not extract text from the PDF file")
        
        # Start background task for processing
        background_tasks.add_task(
            generate_questions_from_pdf,
            pdf_text,
            count,
            grade_level,
            temperature,
            task_id,
            model
        )
        
        return {
            "task_id": task_id,
            "status": "in_progress",
            "message": f"Generation of {count} questions from PDF '{pdf_file.filename}' started"
        }
        
    except Exception as e:
        # Update cache with error
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Error processing PDF file: {str(e)}"
        
        return {
            "task_id": task_id,
            "status": "failed",
            "message": f"Error processing PDF file: {str(e)}"
        }
    
@app.post("/generate-notes-from-pdf", response_model=TaskResponse)
async def create_notes_from_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    notes_type: str = Form("Quick Notes"),
    grade_level: str = Form("elementary"),
    temperature: float = Form(0.2),
    model: str = Form("gemini-1.5-pro-002")
):
    """Generate multiple choice questions from a PDF file"""
    # Create a task ID
    task_id = str(uuid.uuid4())
    
    # Add task to cache
    notes_question_cache[task_id] = {
        "task_id": task_id,
        "status": "in_progress",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
        "pdf_filename": pdf_file.filename
    }
    
    try:
        # Create a temporary directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Save the uploaded PDF temporarily
        pdf_path = f"temp/{task_id}_{pdf_file.filename}"
        with open(pdf_path, "wb") as file:
            shutil.copyfileobj(pdf_file.file, file)
        
        # Extract text from PDF using PyMuPDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Delete the temporary PDF file
        os.remove(pdf_path)
        
        if not pdf_text.strip():
            raise Exception("Could not extract text from the PDF file please try with another pdf")
        
        # Start background task for processing
        background_tasks.add_task(
            generate_notes_from_pdf,
            pdf_text,
            notes_type,
            grade_level,
            temperature,
            task_id,
            model
        )
        
        return {
            "task_id": task_id,
            "status": "in_progress",
            "message": f"Generation of {notes_type} from PDF '{pdf_file.filename}' started"
        }
        
    except Exception as e:
        # Update cache with error
        notes_question_cache[task_id]["status"] = "failed"
        notes_question_cache[task_id]["completed_at"] = datetime.now().isoformat()
        notes_question_cache[task_id]["error"] = f"Error processing PDF file: {str(e)}"
        
        return {
            "task_id": task_id,
            "status": "failed",
            "message": f"Error processing PDF file: {str(e)}"
        }

@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get status of a question generation task"""
    if task_id not in notes_question_cache:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return notes_question_cache[task_id]

@app.get("/questions/{task_id}", response_model=Union[QuestionBank, PDFNotes])
async def get_questions(task_id: str):
    """Get generated questions for a completed task"""
    if task_id not in notes_question_cache:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = notes_question_cache[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task is not completed. Current status: {task['status']}"
        )
    
    return task["result"]

@app.get("/chat/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get history for a specific chat conversation"""
    if conversation_id not in chat_history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"conversation_id": conversation_id, "messages": chat_history[conversation_id]}

@app.get("/")
async def root():
    """API root with basic information"""
    return {
        "message": "MCQ Question Generator API",
        "version": "1.1.0",
        "endpoints": [
            {"path": "/generate", "method": "POST", "description": "Generate new questions"},
            {"path": "/generate-from-pdf", "method": "POST", "description": "Generate questions from PDF content"},
            {"path": "/chat", "method": "POST", "description": "Chat with the LLM model"},
            {"path": "/tasks/{task_id}", "method": "GET", "description": "Check task status"},
            {"path": "/questions/{task_id}", "method": "GET", "description": "Get generated questions"},
            {"path": "/chat/{conversation_id}", "method": "GET", "description": "Get chat conversation history"},
            {"path": "/generate-notes-from-pdf", "method": "POST", "description": "Generate notes from PDF content"}
        ]
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("Starting MCQ Question Generator API v1.1.0 with Google Generative AI")
    
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not set. The API will not function correctly.")
        print("Please set the GOOGLE_API_KEY environment variable before using this application.")
    
    # Create temporary directory for PDF uploads if it doesn't exist
    os.makedirs("temp", exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up temporary files on shutdown"""
    if os.path.exists("temp"):
        try:
            shutil.rmtree("temp")
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
