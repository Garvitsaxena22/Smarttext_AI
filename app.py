import html
import json
import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


# ============================================================
# SMARTTEXT AI — FINAL STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="SmartText AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {"N-gram": BASE_DIR / "smarttext_ngram.pkl", "LSTM": BASE_DIR / "smarttext_lstm.keras", "GRU": BASE_DIR / "smarttext_gru.keras"}
CONFIG_PATH = BASE_DIR / "smarttext_config.json"
VOCAB_PATH = BASE_DIR / "smarttext_vocabulary.json"
DEFAULT_EXAMPLE = "at the end of the"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "editor_text": "",
    "last_predictions": [],
    "prediction_error": "",
    "prediction_count": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PREMIUM UI THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #080d1b;
        --surface: #111a30;
        --surface-2: #17223c;
        --border: #293858;
        --purple: #8b78ff;
        --cyan: #72d8f5;
        --text: #f5f6ff;
        --muted: #9eaccb;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(103, 81, 205, .18), transparent 34%),
            radial-gradient(ellipse at 90% 15%, rgba(42, 145, 190, .12), transparent 30%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 27, .88);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        padding-left: clamp(1.1rem, 3vw, 3rem);
        padding-right: clamp(1.1rem, 3vw, 3rem);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -.4px;
    }

    p, li, label {
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10182c 0%, #0d1426 100%);
        border-right: 1px solid #26324d;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: #2b3753;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.75;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(27px, 4vw, 48px);
        border-radius: 27px;
        border: 1px solid rgba(151, 137, 255, .28);
        background:
            radial-gradient(ellipse at 100% 0%, rgba(105, 88, 214, .24), transparent 43%),
            linear-gradient(125deg, #151c37 0%, #211d42 55%, #13263c 100%);
        box-shadow: 0 18px 55px rgba(0, 0, 0, .18);
        margin-bottom: 27px;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -105px;
        bottom: -145px;
        border: 1px solid rgba(166, 154, 255, .22);
        border-radius: 50%;
        box-shadow:
            0 0 0 25px rgba(166, 154, 255, .035),
            0 0 0 50px rgba(166, 154, 255, .025);
        pointer-events: none;
    }

    .eyebrow {
        color: #b9afff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.5px;
        line-height: 1.8;
        text-transform: uppercase;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(36px, 4.7vw, 61px);
        font-weight: 700;
        letter-spacing: -2.5px;
        line-height: 1.08;
        margin: 18px 0;
        color: #ffffff;
    }

    .gradient-text {
        background: linear-gradient(100deg, #b9aaff 5%, #8bbfff 55%, #73e2f3 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        position: relative;
        z-index: 1;
        color: #c1cbe5;
        max-width: 720px;
        font-size: 15px;
        line-height: 1.9;
        margin-bottom: 16px;
    }

    .pill {
        display: inline-block;
        border: 1px solid rgba(160, 148, 255, .28);
        border-radius: 30px;
        padding: 7px 12px;
        color: #dcd6ff;
        font-size: 11px;
        font-weight: 600;
        margin: 5px 5px 0 0;
        background: rgba(112, 94, 197, .12);
        backdrop-filter: blur(8px);
    }

    .metric-card {
        background: linear-gradient(145deg, #151f37, #11192d);
        border: 1px solid #2c3a5a;
        border-radius: 19px;
        padding: 21px;
        min-height: 105px;
        box-shadow: 0 9px 25px rgba(0, 0, 0, .10);
        transition: border-color .2s ease, transform .2s ease;
    }

    .metric-card:hover {
        border-color: #625b9c;
        transform: translateY(-2px);
    }

    .metric-label {
        color: #a3b2d2;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #f8f8ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(23px, 2.3vw, 30px);
        font-weight: 700;
        letter-spacing: -.7px;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 23px;
        font-weight: 700;
        letter-spacing: -.7px;
        margin-bottom: 7px;
        color: #f6f6ff;
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.8;
    }

    .stTextArea textarea {
        background: linear-gradient(145deg, #111a2e, #0d1527) !important;
        color: #f5f6ff !important;
        border: 1px solid #344361 !important;
        border-radius: 17px !important;
        padding: 17px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        line-height: 1.9 !important;
        caret-color: #a99aff;
        transition: border-color .2s ease, box-shadow .2s ease;
    }

    .stTextArea textarea:focus {
        border-color: #8c7aff !important;
        box-shadow: 0 0 0 1px rgba(140, 122, 255, .35),
                    0 0 24px rgba(119, 99, 238, .08) !important;
    }

    .stTextArea textarea::placeholder {
        color: #68799c !important;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #354464;
        min-height: 43px;
        padding: 8px 14px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 13px;
        background: #151e33;
        color: #e9edff;
        transition: all .2s ease;
    }

    div.stButton > button:hover {
        border-color: #9c8bff;
        color: #ffffff;
        background: #202747;
        box-shadow: 0 5px 18px rgba(95, 75, 190, .13);
        transform: translateY(-1px);
    }

    div.stButton > button:focus:not(:active) {
        border-color: #9c8bff;
        box-shadow: 0 0 0 2px rgba(156, 139, 255, .2);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(105deg, #8068f4, #6258d2);
        border: 1px solid #8b79ff;
        color: #ffffff;
        box-shadow: 0 5px 20px rgba(112, 88, 225, .2);
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(105deg, #927fff, #7468e8);
        border-color: #b3a7ff;
        box-shadow: 0 7px 24px rgba(112, 88, 225, .3);
    }

    .best-prediction {
        background:
            radial-gradient(ellipse at 100% 0%, rgba(112, 216, 245, .09), transparent 55%),
            linear-gradient(130deg, #211e46, #17263d);
        border: 1px solid #615a9e;
        border-radius: 20px;
        padding: 23px;
        margin: 17px 0 15px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, .12);
    }

    .best-word {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(27px, 3vw, 37px);
        font-weight: 700;
        letter-spacing: -.8px;
        line-height: 1.3;
        margin: 8px 0;
        color: #d4cbff;
        overflow-wrap: anywhere;
    }

    .rank-label {
        color: #aeb4e6;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.8px;
    }

    .suggestion {
        background: linear-gradient(130deg, #17223b, #151b32);
        border: 1px solid #303e5f;
        border-radius: 15px;
        padding: 16px 17px;
        margin: 12px 0 8px;
        transition: border-color .2s ease;
    }

    .suggestion:hover {
        border-color: #6c64a8;
    }

    .suggestion-word {
        color: #e0dbff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 700;
        line-height: 1.5;
        margin: 5px 0;
        overflow-wrap: anywhere;
    }

    [data-testid="stSlider"] [role="slider"] {
        border-color: #a99aff !important;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
        border: 1px solid #384663;
    }

    .footer {
        border-top: 1px solid #283551;
        margin-top: 40px;
        padding-top: 22px;
        text-align: center;
        color: #8998b9;
        font-size: 12px;
        line-height: 2;
    }

    .footer strong {
        color: #c5bbff;
        font-weight: 700;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0a1020;
    }

    ::-webkit-scrollbar-thumb {
        background: #344160;
        border-radius: 20px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #6f63b7;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.3rem;
        }

        .hero {
            padding: 26px;
            border-radius: 20px;
        }

        .hero-title {
            letter-spacing: -1.5px;
        }

        .section-heading {
            font-size: 21px;
        }

        .metric-card {
            padding: 15px;
        }
    }

    @media (max-width: 600px) {
        .hero {
            padding: 22px;
        }

        .hero-title {
            font-size: 37px;
        }

        .hero-description {
            font-size: 14px;
        }

        .pill {
            margin-top: 7px;
        }

        .metric-value {
            font-size: 21px;
        }

        .best-prediction {
            padding: 18px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL AND VOCABULARY
# Vocabulary format: {"word": token_id, ...}
# ============================================================

@st.cache_resource
def load_resources(model_name, model_path, vocab_path, model_mtime, vocab_mtime, sequence_length):
    with open(vocab_path, encoding="utf-8") as f:
        word_to_id = json.load(f)
    if word_to_id.get("<PAD>") != 0 or word_to_id.get("<UNK>") != 1 or set(word_to_id.values()) != set(range(len(word_to_id))):
        raise ValueError("Invalid vocabulary mapping.")
    id_to_word = {i: w for w, i in word_to_id.items()}
    if model_name == "N-gram":
        # Only load the trusted local export created by Step 10.
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        required = {"ngram_counts", "bigram_counts", "unigram_counts", "unigram_total", "unigram_top"}
        if not required.issubset(model):
            raise ValueError("Invalid N-gram export.")
    else:
        model = tf.keras.models.load_model(model_path, compile=False)
        layer_type = tf.keras.layers.LSTM if model_name == "LSTM" else tf.keras.layers.GRU
        if not any(isinstance(layer, layer_type) for layer in model.layers):
            raise ValueError("Wrong neural model architecture.")
        if model.output_shape[-1] != len(word_to_id) or model.input_shape[-1] != sequence_length:
            raise ValueError("Model and vocabulary/configuration do not match.")
    return model, word_to_id, id_to_word, sequence_length


def ngram_predict(text, model, word_to_id, id_to_word, top_k):
    tokens = tokenize_text(text)
    if not tokens:
        raise ValueError("Please enter some text before predicting.")
    context = tuple(word_to_id.get(t, 1) for t in tokens[-2:])
    scores = {}
    for counts, weight in [(model["ngram_counts"].get(context), 1.0), (model["bigram_counts"].get(context[-1]), 0.25)]:
        if counts:
            total = sum(counts.values())
            for i, c in counts.items():
                scores[i] = scores.get(i, 0) + weight * c / total
    for i, c in model["unigram_top"][:top_k]:
        scores.setdefault(i, 0)
    for i in scores:
        scores[i] += 0.05 * model["unigram_counts"].get(i, 0) / max(1, model["unigram_total"])
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [{"word": id_to_word[i], "probability": float(score)} for i, score in ranked if i > 1 and i in id_to_word and is_displayable_word(id_to_word[i])][:top_k]


def score_label(item):
    if st.session_state["selected_model"] == "N-gram":
        return f'Ranking score: {item["probability"]:.4f}'
    return f'Model probability: {item["probability"] * 100:.2f}%'


# ============================================================
# TEXT PREPROCESSING AND PREDICTION
# ============================================================

def tokenize_text(text):
    text = text.lower()
    text = text.replace("@-@", "-").replace("@.@", ".").replace("@,@", ",")
    text = re.sub(r"[^a-z0-9\s.,!?;:'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*|[.,!?;:]", text)


def is_displayable_word(word):
    if not isinstance(word, str) or not word.strip():
        return False
    if word.startswith("<") and word.endswith(">"):
        return False
    # Exclude punctuation and numeric-only tokens.
    if not any(char.isalpha() for char in word):
        return False
    return True


def predict_next_words(
    text,
    model,
    word_to_id,
    id_to_word,
    sequence_length,
    top_k=5,
):
    tokens = tokenize_text(text)

    if not tokens:
        raise ValueError("Please enter some text before predicting.")

    unknown_id = word_to_id.get("<UNK>")
    token_ids = []

    for token in tokens:
        if token in word_to_id:
            token_ids.append(word_to_id[token])
        elif unknown_id is not None:
            token_ids.append(unknown_id)

    if not token_ids:
        raise ValueError("No usable tokens were found in your text.")

    token_ids = token_ids[-sequence_length:]

    # Left-pad short inputs with <PAD> ID 0; keep most recent tokens.
    input_ids = np.zeros((1, sequence_length), dtype=np.int32)
    input_ids[0, -len(token_ids):] = token_ids

    if all(token_id == unknown_id for token_id in token_ids):
        raise ValueError("None of the recent tokens are in the trained vocabulary. Try a more common English phrase.")

    # Direct inference avoids model.predict overhead for a single input.
    probabilities = np.asarray(model(input_ids, training=False))[0]
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid probabilities.")

    # Same eligible output classes as Step 9 of the fast notebook.
    allowed = np.array([
        i for i, word in id_to_word.items()
        if i > 1 and is_displayable_word(word)
    ], dtype=np.int32)
    candidate_ids = allowed[np.argsort(-probabilities[allowed], kind="stable")]

    predictions = []
    seen_words = set()

    for token_id in candidate_ids:
        word = id_to_word.get(int(token_id))

        if not is_displayable_word(word):
            continue

        if word.lower() in seen_words:
            continue

        seen_words.add(word.lower())
        predictions.append({
            "word": word,
            "probability": float(probabilities[token_id]),
        })

        if len(predictions) >= top_k:
            break

    if not predictions:
        raise ValueError(
            "No word suggestions were returned. Check your model "
            "and vocabulary files."
        )

    return predictions


# ============================================================
# CALLBACKS
# ============================================================

def invalidate_predictions():
    st.session_state["last_predictions"] = []
    st.session_state["prediction_error"] = ""


def clear_editor():
    st.session_state["editor_text"] = ""
    st.session_state["last_predictions"] = []
    st.session_state["prediction_error"] = ""


def load_example():
    st.session_state["editor_text"] = DEFAULT_EXAMPLE
    st.session_state["last_predictions"] = []
    st.session_state["prediction_error"] = ""


def insert_suggestion(word):
    current = st.session_state.get("editor_text", "").rstrip()

    if current and not current[-1].isspace():
        current += " "

    st.session_state["editor_text"] = current + word + " "
    st.session_state["last_predictions"] = []
    st.session_state["prediction_error"] = ""


def run_prediction():
    try:
        text = st.session_state.get("editor_text", "").strip()

        if not text:
            st.session_state["last_predictions"] = []
            st.session_state["prediction_error"] = (
                "Please enter some text first."
            )
            return

        results = (ngram_predict(text, st.session_state["model"], st.session_state["word_to_id"], st.session_state["id_to_word"], st.session_state["top_k"]) if st.session_state["selected_model"] == "N-gram" else predict_next_words(
            text=text,
            model=st.session_state["model"],
            word_to_id=st.session_state["word_to_id"],
            id_to_word=st.session_state["id_to_word"],
            sequence_length=st.session_state["sequence_length"],
            top_k=st.session_state["top_k"],
        ))

        st.session_state["last_predictions"] = results
        st.session_state["prediction_error"] = ""
        st.session_state["prediction_count"] += 1

    except Exception as error:
        st.session_state["last_predictions"] = []
        st.session_state["prediction_error"] = str(error)


# ============================================================
# SIDEBAR
# ============================================================

try:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
except Exception as error:
    st.error(f"Cannot read Step 10 configuration: {error}")
    st.stop()

with st.sidebar:
    st.markdown("## ✦ SmartText AI")
    st.caption("Intelligent writing assistant")
    st.divider()

    st.markdown("### Model settings")
    options = list(MODEL_PATHS)
    default = config.get("default_model", "N-gram")
    selected = st.selectbox("Prediction model", options, index=options.index(default) if default in options else 0, key="selected_model", on_change=invalidate_predictions)
    performance = config.get("model_performance", {}).get(selected, {})
    for label, key in [("Top-1 test accuracy", "top_1_accuracy"), ("Top-5 test accuracy", "top_5_accuracy")]:
        if key in performance:
            st.metric(label, f"{performance[key]:.2f}%")

    st.slider(
        "Number of suggestions",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="top_k",
        on_change=invalidate_predictions,
    )

    st.markdown("### About the project")
    st.markdown(
        """
        **SmartText AI** predicts the next word using a trained
        statistical or deep learning language model.

        - Natural Language Processing
        - N-gram, LSTM and GRU prediction
        - Ranked word suggestions
        - Interactive writing editor
        """
    )

    st.divider()
    st.caption("Dataset: WikiText-2")
    st.caption("Architecture: trigram backoff" if selected == "N-gram" else f"Architecture: {selected} · 64 units · 48-dimensional embedding")
    st.caption("Fast project · word-only suggestions")


# ============================================================
# RESOURCE INITIALIZATION
# ============================================================

MODEL_PATH = MODEL_PATHS[selected]
if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
    st.error("Model or vocabulary file not found.")
    st.code(
        f"Model: {MODEL_PATH}\nVocabulary: {VOCAB_PATH}",
        language="text",
    )
    st.stop()

try:
    model, word_to_id, id_to_word, sequence_length = load_resources(
        selected,
        str(MODEL_PATH),
        str(VOCAB_PATH),
        MODEL_PATH.stat().st_mtime,
        VOCAB_PATH.stat().st_mtime,
        int(config["sequence_length"]),
    )

    st.session_state["model"] = model
    st.session_state["word_to_id"] = word_to_id
    st.session_state["id_to_word"] = id_to_word
    st.session_state["sequence_length"] = sequence_length

except Exception as error:
    st.error(f"Unable to load SmartText AI: {error}")
    st.info(
        "Verify that the model and vocabulary belong to the same "
        "training run."
    )
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">NLP · DEEP LEARNING · SMART WRITING</div>
        <div class="hero-title">
            Your thoughts,<br>
            <span class="gradient-text">one word ahead.</span>
        </div>
        <div class="hero-description">
            Continue writing with AI-powered next-word suggestions.
            Enter a sentence, explore predictions, and build your next
            thought one word at a time.
        </div>
        <span class="pill">✦ N-gram / LSTM / GRU</span>
        <span class="pill">⌘ NLP Powered</span>
        <span class="pill">⚡ On-demand Predictions</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METRICS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Vocabulary size</div>
            <div class="metric-value">{len(word_to_id):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Context window</div>
            <div class="metric-value">{2 if selected == "N-gram" else sequence_length} tokens</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predictions made</div>
            <div class="metric-value">{st.session_state["prediction_count"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# ============================================================
# EDITOR AND PREDICTIONS
# ============================================================

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown(
        '<div class="section-heading">✍ Your writing space</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">Write a phrase or sentence and ask the model '
        'to suggest what could come next.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    st.text_area(
        "Your text",
        key="editor_text",
        on_change=invalidate_predictions,
        height=220,
        placeholder="Start typing... e.g. at the end of the",
        label_visibility="collapsed",
    )

    col_predict, col_clear, col_example = st.columns([1.3, 0.8, 1.15])

    with col_predict:
        st.button(
            "✦ Predict",
            key="predict_button",
            type="primary",
            on_click=run_prediction,
            use_container_width=True,
        )

    with col_clear:
        st.button(
            "↻ Clear",
            key="clear_button",
            on_click=clear_editor,
            use_container_width=True,
        )

    with col_example:
        st.button(
            "✧ Try example",
            key="example_button",
            on_click=load_example,
            use_container_width=True,
        )

    if st.session_state["prediction_error"]:
        st.error(st.session_state["prediction_error"])


with right_col:
    st.markdown(
        '<div class="section-heading">✦ AI suggestions</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        ('<div class="muted">N-gram ranking scores are not probabilities.</div>' if selected == 'N-gram' else '<div class="muted">Model probabilities are not renormalized after filtering.</div>'),
        unsafe_allow_html=True,
    )

    predictions = st.session_state["last_predictions"]

    if predictions:
        best = predictions[0]

        st.markdown(
            f"""
            <div class="best-prediction">
                <div class="rank-label">TOP PREDICTION</div>
                <div class="best-word">{html.escape(best["word"])}</div>
                <div class="muted">
                    {score_label(best)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "✦ Use top prediction",
            key="use_top_prediction",
            type="primary",
            on_click=insert_suggestion,
            args=(best["word"],),
            use_container_width=True,
        )

        st.write("")
        st.markdown("#### Other possibilities")

        for index, item in enumerate(predictions[1:], start=1):
            safe_word = html.escape(item["word"])
            probability = item["probability"] * 100

            st.markdown(
                f"""
                <div class="suggestion">
                    <div class="rank-label">OPTION {index + 1}</div>
                    <div class="suggestion-word">{safe_word}</div>
                    <div class="muted">
                        {score_label(item)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.button(
                f"＋ Insert {item['word']}",
                key=f"insert_suggestion_{index}",
                on_click=insert_suggestion,
                args=(item["word"],),
                use_container_width=True,
            )

    else:
        st.markdown(
            """
            <div class="suggestion">
                <div style="font-size:30px;">✧</div>
                <div style="font-size:18px;font-weight:600;color:#f5f5ff;">
                    Your next word is waiting
                </div>
                <p class="muted">
                    Enter text on the left and click Predict.
                    Your ranked suggestions will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Try a starter prompt:**")

        starter_prompts = [
            "at the end of the",
            "the film was released in",
            "during the second world",
        ]

        for index, prompt in enumerate(starter_prompts):
            st.button(
                prompt,
                key=f"starter_prompt_{index}",
                on_click=lambda p=prompt: st.session_state.update(
                    editor_text=p,
                    last_predictions=[],
                    prediction_error="",
                ),
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>SmartText AI</strong> · Next Word Prediction System<br>
        Built with Python, Streamlit, TensorFlow and NLP
    </div>
    """,
    unsafe_allow_html=True,
)