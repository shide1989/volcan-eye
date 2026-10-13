from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.get("/predict")
def predict():
    return {"status": "ok"}

@app.get("/volcanoes")
def volcanoes():
    return {"status": "ok"}

@app.get("/volcanoes/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
