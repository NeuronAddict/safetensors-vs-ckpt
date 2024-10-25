import sys

import gradio as gr
from fastai.learner import load_learner, Learner

model = None

def load_file(file):
    global model
    try:
        model = load_learner(file)
        return f'model loaded {model}'
    except Exception as e:
        print(e, file=sys.stderr)
        return 'ERROR LOADING MODEL'

def pred(image):
    try:
        # noinspection PyUnresolvedReferences
        return model.predict(image)  # pyright: ignore [reportOptionalMemberAccess]
    except Exception as e:
        print(e, file=sys.stderr)
        return 'Please load a model and try again.'


if __name__ == "__main__":
    with gr.Blocks(title='Test your model') as demo:
        with gr.Row():
            gr.Interface(load_file, gr.File(), 'text', title="Load a model first")
            gr.Interface(pred, gr.Image(), "text", title="And predict")
        demo.launch()
