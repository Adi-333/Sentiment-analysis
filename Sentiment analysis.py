from transformers import pipeline
import gradio as gr

classifier = pipeline("sentiment-analysis")

def analize(Query):
    res = classifier(Query)
    return (res[0]["label"], res[0]["score"])

iface = gr.Interface(
    fn = analize,
    inputs= ["text"],
    outputs= [gr.Textbox(label="Sentiment"), gr.Number(label="Scores")],

)

iface.launch()
