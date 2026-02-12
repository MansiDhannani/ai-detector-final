from hybrid_ai_detector import detect
import gradio as gr
import os

def gradio_interface(text):
    try:
        result = detect(text)
        return f"Verdict: {result['prediction']} (Confidence: {result['confidence']:.2%})"
    except Exception as e:
        return f"⚠️ Error during analysis: {str(e)}"

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Input Code", lines=10, placeholder="Paste code here..."),
    outputs=gr.Textbox(label="Analysis Result"),
    title="AI Code Detector",
    examples=[
        ["def calculate_sum(a, b):\n    return a + b"],
        ["import torch\nimport torch.nn as nn\n\nclass SimpleModel(nn.Module):\n    def __init__(self)"]
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting Gradio on 0.0.0.0:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )