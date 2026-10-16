import gradio as gr
from fastapi import FastAPI

from api.ui import demo
from api.volcanoes import list_volcanoes

description = """

VolcanEye is the best api to predict if an eruption is about to come, I swear.

"""

app = FastAPI(
    title="VolcanEye",
    description=description,
    summary="Keep an eye on your local volcano.",
    version="0.0.1",
    contact={
        "name": "Julie Cornet",
        "url": "https://www.linkedin.com/in/julie-cornet-1a13401a3/",
        "email": "julie@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "Apache-2.0",
    },
)

@app.get("/")
def read_root():
    return {"status": "ok"}

# SEISMIC

@app.get("/seismic/predict")
def predict():
    return {"prediction": "eruption de ouf"}

# OTHERS (VSO)

# VOLCANOES' DATA

@app.get("/volcanoes")
def volcanoes():
    '''
    The volcanoes shown in the UI, with their past eruption dates and stations as [network, station].
    '''
    return list_volcanoes()

@app.get("/volcanoes/{item_id}")
def read_item(item_id: int):
    '''
    Fetch a volcano by it's ID.

    Args:
        volcano_id: Unique identifier
    '''

    return {"item_id": item_id}

# UI

app = gr.mount_gradio_app(app, demo, path="/ui")
