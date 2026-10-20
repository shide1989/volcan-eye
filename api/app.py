import gradio as gr
from fastapi import FastAPI

from ml_logic.registry import load_model
from ml_logic.model import predict_new

from obspy.clients.fdsn import Client
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

app.state.model = load_model()
client = Client("RESIF", debug=True)

@app.get("/")
def read_root():
    return {"status": "ok"}

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

# SEISMIC
@app.get("/seismic/predict")
def predict():
    label, proba = predict_new(
        model=app.state.model,
        client_name=client,
        network='G',
        station='RER'
        )

    return {
            "prediction": label,
            "proba": proba
        }

# OTHERS (VSO)
# UI

app = gr.mount_gradio_app(app, demo, path="/ui")
