from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import VectorDBManager
import gradio as gr
import json
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.responses import Response, RedirectResponse

# Initialize components
model = OllamaLLM(model="mistral:latest")
db_manager = VectorDBManager()
current_sources = []
current_topic = "restaurant_reviews"  # Default topic

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def load_config():
    """Load configuration from environment variables or .env file"""
    # Try Kubernetes/Docker environment first
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral:latest")
    
    # Fall back to .env file if running locally
    env_path = Path(__file__).parent.parent / "config" / "secrets.env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("Loaded config from secrets.env file")
    else:
        print(f"Warning: No environment config found - using only OS environment variables")
    
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", ollama_base_url),
        "ollama_model": os.getenv("OLLAMA_MODEL", ollama_model)
    }

def rebuild_vector_db(sources):
    """Rebuild the vector database with new sources"""
    global db_manager, current_sources
    try:
        current_sources = sources
        retriever = db_manager.create_or_update_collection(
            topic=current_topic,
            sources=sources,
            metadata={"domain": "restaurant", "type": "reviews"}
        )
        return "Vector database rebuilt successfully!"
    except Exception as e:
        return f"Error rebuilding vector database: {str(e)}"

def query_restaurant(question):
    """Query the restaurant knowledge base"""
    if not question.strip():
        return "Please enter a question about the restaurant."
    
    try:
        retriever = db_manager.get_retriever(current_topic)
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})
        
        # Format the response
        formatted_response = "<div class='response-section response-main'><h2>RESTAURANT REVIEW ANALYSIS</h2>\n\n"
        formatted_response += result
        formatted_response += "</div>"
        
        formatted_response += "\n\n<div class='response-section references-section'><h3>INFORMATION SOURCES</h3>\n\n"
        for source in current_sources:
            formatted_response += f"<div class='reference-item'>Loaded from: {source}</div>\n\n"
        formatted_response += "</div>"
        
        return formatted_response
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Rest of the file remains unchanged (Gradio interface and FastAPI setup)
with gr.Blocks(css="""
    :root {
        --text-color: #000000;
        --background-color: #f5f7f9;
        --primary-color: #1b5e20;
        --secondary-color: #2e7d32;
        --accent-color: #4caf50;
        --section-bg-light: rgba(220, 237, 220, 0.7);
        --section-bg-dark: rgba(35, 70, 35, 0.7);
        --border-radius: 8px;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #ffffff;
            --background-color: #1c2021;
        }
    }
    
    /* Hide Gradio branding */
    footer {
        display: none !important;
    }
    .gradio-container .gradio-footer {
        visibility: hidden !important;
    }
    
    .contain {
        width: 100% !important;
        margin: 0;
        padding: 0;
    }
    .header {
        background-color: var(--primary-color);
        color: white;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .query-box {
        background-color: transparent;
        padding: 15px;
        border: none;
    }
    .response-box {
        background-color: transparent;
        padding: 15px;
        border: none;
        color: var(--text-color);
    }
    .submit-btn {
        background-color: var(--secondary-color) !important;
        width: 100% !important;
        display: block !important;
        margin-bottom: 10px;
    }
    .feedback-section {
        margin-top: 20px;
        background-color: transparent;
        padding: 15px;
        border: none;
    }
    #response-content {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    .examples-parent {
        padding: 10px;
    }
    
    /* Response section styling */
    .response-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: var(--border-radius);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .response-section h2, 
    .response-section h3 {
        margin-top: 0;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.3);
        font-weight: 600;
        color: #e65100;
    }
    
    .response-main {
        background-color: var(--section-bg-light);
    }
    
    .references-section {
        background-color: var(--section-bg-light);
    }
    
    .reference-item {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        background-color: rgba(0, 0, 0, 0.05);
    }
    
    @media (prefers-color-scheme: dark) {
        .response-main, 
        .references-section {
            background-color: var(--section-bg-dark);
        }
        
        .reference-item {
            background-color: rgba(255, 255, 255, 0.05);
        }
    }
    
    .loading-animation {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,0,0,.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
        vertical-align: middle;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @media (prefers-color-scheme: dark) {
        .loading-animation {
            border-color: rgba(255,255,255,.3);
            border-top-color: var(--accent-color);
        }
    }
""", title="Restaurant Review Analyzer", theme="default") as demo:
    
    with gr.Column(elem_classes="contain"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# Restaurant Review Analyzer")
            gr.Markdown("Analyze and query customer reviews")
        
        with gr.Column(elem_classes="query-box"):
            with gr.Row():
                file_input = gr.File(
                    label="Upload Review Files",
                    file_types=[".pdf", ".json", ".csv"],
                    file_count="multiple"
                )
                rebuild_btn = gr.Button("Rebuild Vector DB", variant="primary")
            rebuild_status = gr.Textbox(label="Vector DB Status", interactive=False)
            
            question = gr.Textbox(
                placeholder="Ask about customer reviews...",
                label="Restaurant Review Query",
                lines=5
            )
            submit_btn = gr.Button("Submit", elem_classes="submit-btn", variant="primary")
            loading_indicator = gr.HTML("""
                <div id="loading-spinner" style="display:none; margin-top:10px; text-align:center;">
                    <div class="loading-animation"></div>
                    <span style="margin-left:10px; vertical-align:middle;">Processing query...</span>
                </div>
            """)
        
        with gr.Column(elem_classes="response-box"):
            response = gr.HTML(elem_id="response-content")
        
        # Examples
        gr.Examples(
            examples=[
                "What are customers saying about our pizza?",
                "What are the most common complaints?",
                "What do customers like about our service?",
                "How are our dessert items being reviewed?",
                "What suggestions do customers have for improvement?"
            ],
            inputs=question
        )
    
    # Connect the components
    rebuild_btn.click(
        fn=lambda files: rebuild_vector_db([f.name for f in files]),
        inputs=file_input,
        outputs=rebuild_status
    )
    
    submit_btn.click(
        fn=query_restaurant,
        inputs=question,
        outputs=response
    )
    
    # Add JavaScript for loading indicator
    demo.load(None, None, None, js="""
        function() {
            const submitBtn = document.querySelector("button.submit-btn");
            const spinner = document.getElementById("loading-spinner");
            
            if (submitBtn && spinner) {
                submitBtn.addEventListener("click", function() {
                    spinner.style.display = "inline-flex";
                });
                
                // Hide spinner when new content loads
                const observer = new MutationObserver((mutations) => {
                    const responseArea = document.getElementById("response-content");
                    if (responseArea && responseArea.textContent.trim().length > 0) {
                        spinner.style.display = "none";
                    }
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    characterData: true
                });
            }
        }
    """)

# Launch the app
if __name__ == "__main__":
    # Create FastAPI app
    app = FastAPI()
    
    # Add health check endpoints
    @app.get("/health/ping")
    def health_check_ping():
        return Response("1", media_type="text/plain")

    @app.get("/health")
    def health_check():
        return Response("1", media_type="text/plain")

    # Add index route that redirects to the root path
    @app.get("/index")
    async def index():
        return RedirectResponse(url="")
    
    # Mount Gradio app to the FastAPI app
    from gradio.routes import mount_gradio_app
    app = mount_gradio_app(app, demo, path="")
    
    # Get port from environment variable or default to 7860
    port = int(os.getenv("PORT", 7860))
    
    # Launch with appropriate settings
    import uvicorn
    if os.getenv('location') == "local":
        uvicorn.run(app, host="127.0.0.1", port=port)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)