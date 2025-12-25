
import gradio as gr
import plotly.graph_objects as go
import time

def make_plot(n):
    fig = go.Figure(data=[go.Scatter3d(x=[0, n], y=[0, n], z=[0, n], mode='markers+lines')])
    fig.update_layout(title=f"Test Plot {n}")
    return fig

with gr.Blocks() as demo:
    gr.Markdown("# Ultra Simple Plotly Test")
    with gr.Row():
        btn1 = gr.Button("Plot 1")
        btn2 = gr.Button("Plot 2")
    plot = gr.Plot()
    
    btn1.click(fn=lambda: make_plot(1), outputs=plot)
    btn2.click(fn=lambda: make_plot(2), outputs=plot)

demo.launch(server_port=7869)
