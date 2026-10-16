"""Gradio screen mounted at /ui: pick a volcano, then predict from now -> T+8 h or replay a past eruption."""

import datetime
import re
import time

import gradio as gr
import plotly.graph_objects as go
import psycopg

from api.predict import predict_proba
from api.volcanoes import list_volcanoes

PREDICT, REPLAY = "Predict (now → T+8 h)", "Replay a past eruption"
DEFAULT_VOLCANO = 233020  # Piton de la Fournaise
MIN_LOADING_S = 0.5  # show the loading state at least this long, even when the answer is faster


def level(p):
    """Result badge: (text, color)."""
    if p < 0.3:
        return "calm", "#16a34a"
    if p <= 0.7:
        return "restless", "#d97706"
    return "eruption", "#dc2626"


def db_error(e):
    return gr.Error(f"Could not read the volcano database ({type(e).__name__}). "
                    "Check that DATABASE_URL is set and that the Supabase project is up.")


def volcano(volcano_id):
    return next(v for v in list_volcanoes() if v["id"] == volcano_id)


def volcano_map(volcano_id):
    volcanoes = list_volcanoes()
    selected = volcano(volcano_id)
    fig = go.Figure([
        go.Scattergeo(lat=[v["lat"] for v in volcanoes], lon=[v["lon"] for v in volcanoes],
                      text=[v["name"] for v in volcanoes], hoverinfo="text", marker=dict(size=6, color="#9ca3af")),
        go.Scattergeo(lat=[selected["lat"]], lon=[selected["lon"]], text=[selected["name"]], hoverinfo="text",
                      marker=dict(size=14, color="#dc2626", symbol="triangle-up")),
    ])
    fig.update_geos(projection_type="natural earth", projection_rotation_lon=selected["lon"], showcountries=True)
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def info_card(v):
    summary = " ".join(re.split(r"(?<=[.!?])\s+", v["gvp_summary"] or "")[:2])
    elevation = f"{v['elevation_m']:.0f} m" if v["elevation_m"] is not None else "unknown"
    return (f"### {v['name']}\n**Country** {v['country']} · **Type** {v['primary_volcano_type'] or 'unknown'} · "
            f"**Elevation** {elevation}\n\n{summary}")


def on_load():
    try:
        volcanoes = list_volcanoes()
    except (KeyError, psycopg.Error) as e:
        raise db_error(e)
    ids = [v["id"] for v in volcanoes]
    return gr.update(choices=[(f"{v['name']} ({v['country']})", v["id"]) for v in volcanoes],
                     value=DEFAULT_VOLCANO if DEFAULT_VOLCANO in ids else ids[0])


def on_volcano(volcano_id):
    v = volcano(volcano_id)
    return (
        volcano_map(volcano_id),
        info_card(v),
        gr.update(choices=v["eruptions"], value=v["eruptions"][0] if v["eruptions"] else None),
        gr.update(choices=sorted({network for network, _ in v["stations"]}), value=None),
        gr.update(choices=[], value=None),
        None,  # clear the previous result
        "",
    )


def on_mode(mode):
    replay = mode == REPLAY
    return (gr.update(visible=replay), gr.update(visible=not replay),
            gr.update(value="Replay" if replay else "Predict"))


def on_network(volcano_id, network):
    stations = sorted(s for n, s in volcano(volcano_id)["stations"] if n == network)
    return gr.update(choices=stations, value=None)


def predict(volcano_id, mode, date, network, station):
    if mode == REPLAY and not date:
        raise gr.Error("Pick an eruption date to replay.")
    day = datetime.date.fromisoformat(date) if mode == REPLAY else None
    try:
        return predict_proba(volcano_id, day, network or None, station or None)
    except NotImplementedError:
        raise gr.Error("Predicting from now is not available yet: the model is not plugged in. "
                       "Use 'Replay a past eruption' meanwhile.")
    except ValueError as e:
        raise gr.Error(str(e))
    except (KeyError, psycopg.Error) as e:
        raise db_error(e)


def on_run(volcano_id, mode, date, network, station):
    start = time.monotonic()
    try:
        p = predict(volcano_id, mode, date, network, station)
    finally:
        time.sleep(max(0.0, MIN_LOADING_S - (time.monotonic() - start)))
    text, color = level(p)
    badge = (f'<span style="background:{color};color:white;padding:2px 12px;border-radius:999px;'
             f'font-weight:600">{text}</span>')
    return {"eruption": p}, badge


with gr.Blocks(title="Volcan Eye") as demo:
    gr.Markdown("# Volcan Eye\nProbability of an eruption at T+8 h for one volcano.")
    with gr.Row():
        with gr.Column():
            volcano_plot = gr.Plot(show_label=False)
        with gr.Column():
            volcano_input = gr.Dropdown(label="Volcano", choices=[])
            info = gr.Markdown()
            mode_input = gr.Radio([PREDICT, REPLAY], value=PREDICT, label="Mode")
            date_input = gr.Dropdown(label="Eruption date", choices=[], visible=False)
            with gr.Accordion("Advanced", open=False) as advanced:
                network_input = gr.Dropdown(label="Network", choices=[])
                station_input = gr.Dropdown(label="Station", choices=[])
            button = gr.Button("Predict", variant="primary")
            result = gr.Label(label="Probability of an eruption at T+8 h")
            badge = gr.HTML()

    demo.load(on_load, outputs=volcano_input)
    volcano_input.change(on_volcano, volcano_input,
                         [volcano_plot, info, date_input, network_input, station_input, result, badge])
    mode_input.change(on_mode, mode_input, [date_input, advanced, button])
    network_input.change(on_network, [volcano_input, network_input], station_input)
    button.click(on_run, [volcano_input, mode_input, date_input, network_input, station_input], [result, badge],
                 api_name="predict")
