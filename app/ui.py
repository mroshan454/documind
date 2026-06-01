import requests 
import gradio as gr 
from pathlib import Path 

API_BASE = "http://localhost:8000"

def fetch_status():
    resp = requests.get(API_BASE)
    data = resp.json()
    count = data["stats"]["total_chunks"]
    return f"{count} chunks indexed"

def call_query(question,k):
    resp = requests.post(f"{API_BASE}/query",json={"question":question, "k":k})
    query_data = resp.json()
    answer = query_data["answer"]
    all_sources = query_data["sources"]
    output = "" 
    for item in all_sources:
        output += (f"Source {item.get('source','')} . {item.get('page','')}. {item.get('score','')}")
        output += (f"> \"{item.get('text','')}\"\n")
    
    return answer , output


def call_ingest(file_obj):
    file_path = file_obj.name 

    filename = Path(file_path).name 

    with open(file_path , "rb") as f: 
        files = {"file": (filename, f, "application/pdf")}
        
        response = requests.post(f"{API_BASE}/ingest",files=files)
        

        response_data = response.json()
        
        msg = (f"Successfully Ingested {response_data.get('filename')}!\n"
               f"Processed {response_data.get('pages', 0)} pages into "
               f"{response_data.get('chunks', 0)} chunks." )
          

        return (msg,fetch_status())
    
    
    
def build_gradio_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="emerald", neutral_hue="zinc")) as demo:
    # HEADER
        gr.Markdown(
            "# DocuMind\n"
            "Upload a PDF, ask questions, get answers grounded in the document — "
            "with the exact source chunks shown."
        )
        status_md = gr.Markdown(value="📚 Loading corpus status…")
 
        # INGEST SECTION
        with gr.Group():
            gr.Markdown("### 1. Upload a PDF")
            with gr.Row():
                file_input = gr.File(
                    label="Drop a PDF here",
                    file_types=[".pdf"],
                )
                ingest_btn = gr.Button("Ingest", variant="primary")
            ingest_msg = gr.Markdown()
 
        # QUERY SECTION
        with gr.Group():
            gr.Markdown("### 2. Ask a question")
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is the Tidepod-3's price?",
                lines=2,
            )
            with gr.Row():
                k_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Top-k chunks to retrieve",
                )
                ask_btn = gr.Button("Ask", variant="primary")
 
            answer_box = gr.Markdown(label="Answer")
 
            # Collapsible sources — transparency without cluttering the page
            with gr.Accordion("🔎 Retrieved sources (click to expand)", open=False):
                sources_box = gr.Markdown()
 
        # WIRING
        ingest_btn.click(
            fn=call_ingest,
            inputs=[file_input],
            outputs=[ingest_msg, status_md],   # order matches call_ingest's return tuple
        )
        ask_btn.click(
            fn=call_query,
            inputs=[question_box, k_slider],   # order matches call_query's params
            outputs=[answer_box, sources_box], # order matches call_query's return tuple
        )
        demo.load(fn=fetch_status, inputs=None, outputs=status_md)
    return demo