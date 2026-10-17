from itertools import islice
import streamlit as st
from datasets import load_dataset, Audio

NAME = "NandemoGHS/Japanese-Eroge-Voice-V2"

st.set_page_config(page_title="Local speech dataset explorer", layout="wide")
st.title("Speech dataset explorer (local research use only)")
st.caption("Runs on this computer only. Do not deploy or redistribute the audio.")

n = st.sidebar.slider("Clips to load", 5, 50, 10)
skip = st.sidebar.number_input("Skip first N clips", 0, 100000, 0, step=10)

@st.cache_data(show_spinner="Loading clips...")
def load_clips(n, skip):
    ds = load_dataset(NAME, split="train", streaming=True)
    feats = ds.features or {}
    audio_col = next((k for k, v in feats.items() if isinstance(v, Audio)), "audio")
    ds = ds.cast_column(audio_col, Audio(decode=False))
    rows = list(islice(ds.skip(skip), n))
    return audio_col, rows

audio_col, rows = load_clips(n, skip)
st.write("Columns:", [k for k in rows[0].keys()])

for i, ex in enumerate(rows):
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**#{skip + i}** · speaker: `{ex.get('speaker')}`")
            st.write(ex.get("text"))
        with c2:
            data = ex[audio_col].get("bytes")
            if data:
                st.audio(data)
            else:
                st.caption("No audio bytes for this row.")
