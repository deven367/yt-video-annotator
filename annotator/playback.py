import streamlit as st


def playback_state(key, source):
    """Keep a player's seek position tied to its current source/model."""
    state = st.session_state.get(key)
    if state is None or state["source"] != source:
        state = {
            "source": source,
            "start_time": 0,
            "revision": state["revision"] + 1 if state else 0,
            "autoplay": False,
        }
        st.session_state[key] = state
    return state


def seek_to(key, source, start_time):
    state = playback_state(key, source)
    st.session_state[key] = {
        "source": source,
        "start_time": start_time,
        "revision": state["revision"] + 1,
        "autoplay": True,
    }
