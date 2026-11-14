"""Streamlit-App. Start im Repo-Hauptordner:  streamlit run app/app.py"""
import sys
from pathlib import Path

# Repo-Hauptordner in den Suchpfad, damit "from src import ..." funktioniert
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Dancing with the SARs", layout="wide")
st.title("Dancing with the SARs")

# TODO: Seitenleiste (Ereignis, Zeitpunkte, Schwellenwert)
# TODO: Vorher/Nachher-Layer aus src.flood einbinden

m = folium.Map(location=[0, 0], zoom_start=2)
st_folium(m, height=600)
