# app.py — Streamlit app with robust HF loader + full preprocessing for inputs (text & YouTube comments)
import os
import re
import html
import string
import time
import traceback
import io
from typing import List, Optional

import streamlit as st
import pandas as pd
import torch
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import list_repo_files, hf_hub_download

# --- optional text libraries (Sastrawi) with graceful fallback ---
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _SASTRAWI_AVAILABLE = True
except Exception:
    _SASTRAWI_AVAILABLE = False

# ---------------- Config ----------------
DEFAULT_REPO = "yossss90/indobert-imbalance-2"  # replace with your HF repo
st.set_page_config(page_title="IndoBERT Classifier", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

# ---------------- Preprocessing utilities (copied & adapted) ----------------
FULL_UNICODE_NORMALIZATION_MAP = {
    # Full-width
    'Ａ':'A', 'Ｂ':'B', 'Ｃ':'C', 'Ｄ':'D', 'Ｅ':'E', 'Ｆ':'F', 'Ｇ':'G', 'Ｈ':'H', 'Ｉ':'I', 'Ｊ':'J', 'Ｋ':'K', 'Ｌ':'L', 'Ｍ':'M', 'Ｎ':'N', 'Ｏ':'O', 'Ｐ':'P', 'Ｑ':'Q', 'Ｒ':'R', 'Ｓ':'S', 'Ｔ':'T', 'Ｕ':'U', 'Ｖ':'V', 'Ｗ':'W', 'Ｘ':'X', 'Ｙ':'Y', 'Ｚ':'Z',
    'ａ':'a', 'ｂ':'b', 'ｃ':'c', 'ｄ':'d', 'ｅ':'e', 'ｆ':'f', 'ｇ':'g', 'ｈ':'h', 'ｉ':'i', 'ｊ':'j', 'ｋ':'k', 'ｌ':'l', 'ｍ':'m', 'ｎ':'n', 'ｏ':'o', 'ｐ':'p', 'ｑ':'q', 'ｒ':'r', 'ｓ':'s', 'ｔ':'t', 'ｕ':'u', 'ｖ':'v', 'ｗ':'w', 'ｘ':'x', 'ｙ':'y', 'ｚ':'z',
    '０':'0', '１':'1', '２':'2', '３':'3', '４':'4', '５':'5', '６':'6', '７':'7', '８':'8', '９':'9',

    # Double-Struck
    '𝔸':'A', '𝔹':'B', 'ℂ':'C', '𝔻':'D', '𝔼':'E', '𝔽':'F', '𝔾':'G', 'ℍ':'H', '𝕀':'I', '𝕁':'J', '𝕂':'K', '𝕃':'L', '𝕄':'M', 'ℕ':'N', '𝕆':'O', 'ℙ':'P', 'ℚ':'Q', 'ℝ':'R', '𝕊':'S', '𝕋':'T', '𝕌':'U', '𝕍':'V', '𝕎':'W', '𝕏':'X', '𝕐':'Y', 'ℤ':'Z',
    '𝕒':'a', '𝕓':'b', '𝕔':'c', '𝕕':'d', '𝕖':'e', '𝕗':'f', '𝕘':'g', '𝕙':'h', '𝕚':'i', '𝕛':'j', '𝕜':'k', '𝕝':'l', '𝕞':'m', '𝕟':'n', '𝕠':'o', '𝕡':'p', '𝕢':'q', '𝕣':'r', '𝕤':'s', '𝕥':'t', '𝕦':'u', '𝕧':'v', '𝕨':'w', '𝕩':'x', '𝕪':'y', '𝕫':'z',

    # Mathematical Bold
    '𝐀':'A', '𝐁':'B', '𝐂':'C', '𝐃':'D', '𝐄':'E', '𝐅':'F', '𝐆':'G', '𝐇':'H', '𝐈':'I', '𝐉':'J', '𝐊':'K', '𝐋':'L', '𝐌':'M', '𝐍':'N', '𝐎':'O', '𝐏':'P', '𝐐':'Q', '𝐑':'R', '𝐒':'S', '𝐓':'T', '𝐔':'U', '𝐕':'V', '𝐖':'W', '𝐗':'X', '𝐘':'Y', '𝐙':'Z',
    '𝐚':'a', '𝐛':'b', '𝐜':'c', '𝐝':'d', 'ｅ':'e', '𝐟':'f', 'ｇ':'g', '𝐡':'h', '𝐢':'i', '𝐣':'j', '𝐤':'k', '𝐥':'l', '𝐦':'m', '𝐧':'n', '𝐨':'o', '𝐩':'p', '𝐪':'q', '𝐫':'r', '𝐬':'s', '𝐭':'t', '𝐮':'u', '𝐯':'v', '𝐰':'w', '𝐱':'x', '𝐲':'y', '𝐳':'z',
    '𝟎':'0', '𝟏':'1', '𝟐':'2', '𝟑':'3', '𝟒':'4', '𝟓':'5', '𝟔':'6', '𝟕':'7', '𝟖':'8', '𝟗':'9',

    # Sans-Serif Bold Italic
    '𝘼':'A', '𝘽':'B', '𝘾':'C', '𝘿':'D', '𝙀':'E', '𝙁':'F', '𝙂':'G', '𝙃':'H', '𝙄':'I', '𝙅':'J', '𝙆':'K', '𝙇':'L', '𝙈':'M', '𝙉':'N', '𝙊':'O', '𝙋':'P', '𝙌':'Q', '𝙍':'R', '𝙎':'S', '𝙏':'T', '𝙐':'U', '𝙑':'V', '𝙒':'W', '𝙓':'X', '𝙔':'Y', '𝙕':'Z',
    '𝙖':'a', '𝙗':'b', '𝙘':'c', '𝙙':'d', '𝙚':'e', '𝙛':'f', '𝙜':'g', '𝙝':'h', '𝙞':'i', '𝙟':'j', '𝙠':'k', '𝙡':'l', '𝙢':'m', '𝙣':'n', '𝙤':'o', '𝙥':'p', '𝙦':'q', '𝙧':'r', '𝙨':'s', '𝙩':'t', '𝙪':'u', '𝙫':'v', '𝙬':'w', '𝙭':'x', '𝙮':'y', '𝙯':'z',

    # Sans-Serif Bold
    '𝗔':'A', '𝗕':'B', '𝗖':'C', '𝗗':'D', '𝗘':'E', '𝗙':'F', '𝗚':'G', '𝗛':'H', '𝗜':'I', '𝗝':'J', '𝗞':'K', '𝗟':'L', '𝗠':'M', '𝗡':'N', '𝗢':'O', '𝗣':'P', '𝗤':'Q', '𝗥':'R', '𝗦':'S', '𝗧':'T', '𝗨':'U', '𝗩':'V', '𝗪':'W', '𝗫':'X', '𝗬':'Y', '𝗭':'Z',
    '𝗮':'a', '𝗯':'b', '𝗰':'c', '𝗱':'d', '𝗲':'e', '𝗳':'f', '𝗴':'g', '𝗵':'h', '𝗶':'i', '𝗷':'j', '𝗸':'k', '𝗹':'l', '𝗺':'m', '𝗻':'n', '𝗼':'o', '𝗽':'p', '𝗾':'q', '𝗿':'r', '𝘀':'s', '𝘁':'t', '𝘂':'u', '𝘃':'v', '𝘄':'w', '𝘅':'x', '𝘆':'y', '𝘇':'z',
    '𝟬':'0', '𝟭':'1', '𝟮':'2', '𝟯':'3', '𝟰':'4', '𝟱':'5', '𝟲':'6', '𝟳':'7', '𝟴':'8', '𝟵':'9',

    # Monospace
    '𝙰':'A', '𝙱':'B', '𝙲':'C', '𝙳':'D', '𝙴':'E', '𝗙':'F', '𝙶':'G', '𝙷':'H', '𝙸':'I', '𝙹':'J', '𝙺':'K', '𝙻':'L', '𝙼':'M', '𝙽':'N', '𝙾':'O', '𝙿':'P', '𝚀':'Q', '𝚁':'R', '𝚂':'S', '𝚃':'T', '𝚄':'U', '𝚅':'V', '𝚆':'W', '𝚇':'X', '𝚈':'Y', '𝚉':'Z',
    '𝚊':'a', '𝚋':'b', '𝚌':'c', '𝚍':'d', '𝚎':'e', '𝚏':'f', '𝚐':'g', '𝚑':'h', '𝚒':'i', '𝚓':'j', '𝚔':'k', '𝚕':'l', '𝚖':'m', '𝚗':'n', '𝚘':'o', '𝚙':'p', '𝚚':'q', '𝚛':'r', '𝚜':'s', '𝚝':'t', '𝚞':'u', '𝘃':'v', '𝚠':'w', '𝚡':'x', '𝚢':'y', '𝚣':'z',
    '𝟶':'0', '𝟷':'1', '𝟸':'2', '𝟹':'3', '𝟺':'4', '𝟻':'5', '𝟼':'6', '𝟽':'7', '𝟾':'8', '𝟿':'9',

    # Fraktur / Gothic
    '𝕬':'A', '𝕭':'B', '𝕮':'C', '𝕯':'D', '𝕰':'E', '𝕱':'F', '𝕲':'G', '𝕳':'H', '𝕴':'I', '𝕵':'J', '𝕶':'K', '𝕷':'L', '𝕸':'M', '𝕹':'N', '𝕺':'O', '𝕻':'P', '𝕼':'Q', '𝕽':'R', '𝕾':'S', '𝕿':'T', '𝖀':'U', '𝖁':'V', '𝖂':'W', '𝖃':'X', '𝖄':'Y', '𝖅':'Z',
    '𝖆':'a', '𝖇':'b', '𝖈':'c', '𝖉':'d', '𝖊':'e', '𝖋':'f', '𝖌':'g', '𝖍':'h', '𝖎':'i', '𝖏':'j', '𝖐':'k', '𝖑':'l', '𝖒':'m', '𝖓':'n', '𝖔':'o', '𝖕':'p', '𝖖':'q', '𝖗':'r', '𝖘':'s', '𝖙':'t', '𝖚':'u', '𝖛':'v', '𝖜':'w', '𝖝':'x', '𝖞':'y', '𝖟':'z',

    # Enclosed Alphanumerics
    'Ⓐ':'A', 'Ⓑ':'B', 'Ⓒ':'C', 'Ⓓ':'D', 'Ⓔ':'E', 'Ⓕ':'F', 'Ⓖ':'G', 'Ⓗ':'H', 'Ⓘ':'I', 'Ⓙ':'J', 'Ⓚ':'K', 'Ⓛ':'L', 'Ⓜ':'M', 'Ⓝ':'N', 'Ⓞ':'O', 'Ⓟ':'P', 'Ⓠ':'Q', 'Ⓡ':'R', 'Ⓢ':'S', 'Ⓣ':'T', 'Ⓤ':'U', 'Ⓥ':'V', 'Ⓦ':'W', 'Ⓧ':'X', 'Ⓨ':'Y', 'Ⓩ':'Z',
    'ⓐ':'a', 'ⓑ':'b', 'ⓒ':'c', 'ⓓ':'d', 'ⓔ':'e', 'ⓕ':'f', 'ⓖ':'g', 'ⓗ':'h', 'ⓘ':'i', 'ⓙ':'j', 'ⓚ':'k', 'ⓛ':'l', 'ⓜ':'m', 'ⓝ':'n', 'ⓞ':'o', 'ⓟ':'p', 'ⓠ':'q', 'ⓡ':'r', 'ⓢ':'s', 'ⓣ':'t', 'ⓤ':'u', 'ⓥ':'v', 'ⓦ':'w', 'ⓧ':'x', 'ⓨ':'y', 'ⓩ':'z',
    '🅰':'A', '🅱':'B', '🅲':'C', '🅳':'D', '🅴':'E', '🅵':'F', '🅶':'G', '🅷':'H', '🅸':'I', '🅹':'J', '🅺':'K', '🅻':'L', '🅼':'M', '🅽':'N', '🅾':'O', '🅿':'P', '🆀':'Q', '🆁':'R', '🆂':'S', '🆃':'T', '🆄':'U', '🆅':'V', '🆆':'W', '🆇':'X', '🆈':'Y', '🆉':'Z',
    '🄿':'P', '🄾':'O', '🄽':'N', '🄼':'M', '🄻':'L', '🄺':'K', '🄹':'J', '🄸':'I', '🄷':'H', '🄶':'G', '🄵':'F', '🄴':'E', '🄳':'D', '🄲':'C', '🄱':'B', '🄰':'A',
    '🅀':'Q', '🅁':'R', '🅂':'S', '🅃':'T', '🅄':'U', '🅅':'V', '🅆':'W', '🅇':'X', '🅈':'Y', '🅉':'Z',

    # Other specific characters
    'ڛ': 'S', '𛍃': 'A', '𛍅': 'G', '𛍄': 'A', '𝅙': 'A', '𛌷': 'R', '𛌺': 'D',
    'ᑭ': 'P', 'ᖇ': 'R', 'ᗷ': 'B',
}

MULTI_CHAR_NORMALIZATION_MAP = {
    '0️⃣': '0', '1️⃣': '1', '2️⃣': '2', '3️⃣': '3', '4️⃣': '4', '5️⃣': '5', '6️⃣': '6', '7️⃣': '7', '8️⃣': '8', '9️⃣': '9',
    '❶': '1', '❷': '2', '❸': '3', '❹': '4', '❺': '5', '❻': '6', '❼': '7', '❽': '8', '❾': '9',
}

def normalize_and_clean_styles(text: str) -> str:
    for old, new in MULTI_CHAR_NORMALIZATION_MAP.items():
        text = text.replace(old, new)
    diacritic_stripper = re.compile(r"[\u0300-\u036f\u0483-\u0489\u200b-\u200f\u20d0-\u20ff\ufe0e\ufe0f]")
    text = diacritic_stripper.sub('', text)
    trans_table = str.maketrans(FULL_UNICODE_NORMALIZATION_MAP)
    text = text.translate(trans_table)
    return text

def clean_text_modified(text: str) -> str:
    text = str(text)
    text = re.sub(r'<a[^>]*>.*?</a>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    url_pattern = re.compile(r'(?:https?://|www\.)\S+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/\S*)?')
    text = url_pattern.sub(' ', text)
    text = normalize_and_clean_styles(text)
    text = html.unescape(text)
    punc_to_remove = string.punctuation.replace('-', '')
    pattern = r'[' + re.escape(punc_to_remove) + r']'
    text = re.sub(pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_stemmer(text: str, stemmer) -> str:
    new_tokens = []
    for token in text.split():
        if token.isalpha():
            try:
                new_tokens.append(stemmer.stem(token))
            except Exception:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

if _SASTRAWI_AVAILABLE:
    try:
        stop_factory = StopWordRemoverFactory()
        stop_remover = stop_factory.create_stop_word_remover()
        stem_factory = StemmerFactory()
        stemmer = stem_factory.create_stemmer()
    except Exception:
        stop_remover = None
        stemmer = None
else:
    stop_remover = None
    stemmer = None

def preprocess_text_full(text: str) -> str:
    t = clean_text_modified(text)
    t = t.lower()
    if stop_remover is not None:
        try:
            t = stop_remover.remove(t)
        except Exception:
            pass
    if stemmer is not None:
        try:
            t = safe_stemmer(t, stemmer)
        except Exception:
            pass
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---------------- Robust HF loader ----------------
@st.cache_resource(show_spinner=False)
def load_pipeline_hf(repo_id: str, device_choice: str = "auto"):
    if device_choice == "cpu":
        device = -1
    elif device_choice == "gpu":
        device = 0
    else:
        device = 0 if torch.cuda.is_available() else -1

    hf_token = None
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    kwargs = {"token": hf_token} if hf_token else {}

    try:
        tok = AutoTokenizer.from_pretrained(repo_id, subfolder="model", **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder="model", **kwargs)
        pipe = pipeline("text-classification", model=model, tokenizer=tok, top_k=None, device=device)
        return pipe, device
    except Exception as e:
        try:
            tok = AutoTokenizer.from_pretrained(repo_id, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(repo_id, **kwargs)
            pipe = pipeline("text-classification", model=model, tokenizer=tok, top_k=None, device=device)
            return pipe, device
        except Exception as e2:
            raise RuntimeError(f"Failed to load model from HF.\nSubfolder error: {e}\nRoot error: {e2}")

# ---------------- Utility helpers ----------------
def get_top_prediction(scores_list):
    best = max(scores_list, key=lambda x: x["score"])
    return best["label"], float(best["score"])

def normalize_label(lbl):
    if isinstance(lbl, str):
        lbl = lbl.replace("LABEL_", "")
    try:
        lbl_int = int(lbl)
    except ValueError:
        return str(lbl)
    mapping = {
        0: "Neutral",
        1: "Toxic",
        2: "Online Gambling"
    }
    return mapping.get(lbl_int, str(lbl_int))

def highlight_classes(row):
    lbl = row['predicted_label']
    if lbl == 'Neutral':
        return ['background-color: rgba(39, 174, 96, 0.2)'] * len(row)
    elif lbl == 'Toxic':
        return ['background-color: rgba(231, 76, 60, 0.2)'] * len(row)
    elif lbl == 'Online Gambling':
        return ['background-color: rgba(230, 126, 34, 0.2)'] * len(row)
    return [''] * len(row)

# ---------------- YouTube helpers ----------------
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_VIDEO_API_URL = "https://www.googleapis.com/youtube/v3/videos"

def extract_video_id(url: str) -> Optional[str]:
    regexes = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"youtube\.com/v/([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?.*v=([A-Za-z0-9_-]{11})"
    ]
    for r in regexes:
        m = re.search(r, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    return None

def fetch_youtube_video_info(video_id: str, api_key: str):
    params = {"part": "snippet", "id": video_id, "key": api_key}
    try:
        resp = requests.get(YOUTUBE_VIDEO_API_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("items"):
                snippet = data["items"][0]["snippet"]
                return {
                    "title": snippet.get("title", "Unknown Title"),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "channel": snippet.get("channelTitle", "")
                }
    except:
        pass
    return None

@st.cache_data(ttl=60*60)
def fetch_youtube_comments(video_id: str, api_key: str, max_comments: int = 200) -> List[str]:
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": api_key,
    }
    nextPageToken = None
    while True:
        if nextPageToken:
            params["pageToken"] = nextPageToken
        resp = requests.get(YOUTUBE_API_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"YouTube API error {resp.status_code}: {resp.text}")
        data = resp.json()
        items = data.get("items", [])
        for it in items:
            try:
                text = it["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(text)
                if len(comments) >= max_comments:
                    return comments[:max_comments]
            except Exception:
                continue
        nextPageToken = data.get("nextPageToken")
        if not nextPageToken:
            break
        time.sleep(0.1)
    return comments

# ---------------- Sidebar UI (Cleaned Up) ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Configure your AI model settings here.")

with st.sidebar.expander("Advanced Technical Settings", expanded=False):
    repo_input = st.text_input("Model Repo / Folder", value=DEFAULT_REPO, help="Hugging Face repo (username/repo) or local path")
    device_opt = st.selectbox("Device", options=["auto", "cpu", "gpu"], index=0, help="auto -> use GPU if available")
    show_raw = st.checkbox("Show raw scores (for debugging)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Developed with ❤️ by Group 4 - Deep Learning - 2025")

# ---------------- Main UI & Header ----------------
st.title("🧪 IndoBERT Text & Comment Classifier")
st.info("Welcome! This AI application detects **Online Gambling (Judol)** and **Toxic** sentences in Indonesian texts. You can classify a single text manually or analyze hundreds of comments from a YouTube video automatically.", icon="ℹ️")

with st.spinner("Loading AI Model (only once)..."):
    try:
        nlp, device_used = load_pipeline_hf(repo_input, device_choice=device_opt)
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Failed to load model from `{repo_input}`:\n{e}")
        st.code(tb)
        st.stop()

# ---------------- TABS Navigation ----------------
tab_single, tab_yt = st.tabs(["📝 Single Text Analysis", "▶️ YouTube Comments Analysis"])

# ---------------- TAB 1: Single Text ----------------
with tab_single:
    st.markdown("### Manual Text Classification")
    
    # Initialize session state for text input to allow "Use example" button to work
    if "single_text_input" not in st.session_state:
        st.session_state.single_text_input = ""

    def set_example_text():
        st.session_state.single_text_input = "Website ini sangat gacor, buruan daftar sekarang juga dan dapatkan bonus deposit 100%!"

    st.button("💡 Use Example Text", on_click=set_example_text)
    
    text = st.text_area("Enter your text below:", key="single_text_input", height=140, placeholder="Type a sentence here...")
    
    if st.button("Predict Text", type="primary"):
        if not text or not text.strip():
            st.warning("Input cannot be empty. Please enter some text.")
        else:
            with st.spinner("Performing preprocessing & inference..."):
                pre = preprocess_text_full(text)
                out = nlp(pre)
                
                scores = out[0] if isinstance(out[0], list) else out
                top_label, top_score = get_top_prediction(scores)
                display_label = normalize_label(top_label)

            # Results Layout
            st.markdown("---")
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.markdown("#### 🔎 Text Details")
                st.write("**Original Text:**")
                st.info(text)
                st.write("**Cleaned (Preprocessed) Text:**")
                st.info(pre)
                
            with res_col2:
                st.markdown("#### 🎯 Final Prediction")
                st.metric(label="Predicted Class", value=f"{display_label}", delta=f"{top_score:.2%} Confidence", delta_color="off")
                
                df_data = [{"label": normalize_label(x["label"]), "score": x["score"]} for x in scores]
                df = pd.DataFrame(df_data).sort_values("score", ascending=False).reset_index(drop=True)
                st.bar_chart(df.set_index("label"))

            if show_raw:
                st.markdown("#### Raw Scores")
                st.json(scores)


# ---------------- TAB 2: YouTube Comments ----------------
with tab_yt:
    st.markdown("### Batch YouTube Comment Analysis")
    
    yt_col1, yt_col2 = st.columns([3, 1])
    with yt_col1:
        youtube_url = st.text_input("Enter YouTube Link (or Video ID):", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with yt_col2:
        max_comments = st.slider("Max Comments", min_value=10, max_value=1000, value=200, step=10)
    
    analyze_btn = st.button("Analyze Comments", type="primary")

    # Initialize session state so results persist across app reruns (e.g. when downloading or filtering)
    if "yt_df_res" not in st.session_state:
        st.session_state.yt_df_res = None
        st.session_state.yt_vid = None
        st.session_state.yt_info = None

    if analyze_btn:
        st.session_state.yt_df_res = None
        
        vid = extract_video_id(youtube_url)
        if not vid:
            st.error("Cannot extract Video ID. Ensure the YouTube URL is correct.")
        else:
            api_key = st.secrets.get("YOUTUBE_API_KEY", os.environ.get("YOUTUBE_API_KEY"))
                
            if not api_key:
                st.error("YouTube API key not found. Set `YOUTUBE_API_KEY` in Streamlit secrets.")
            else:
                st.session_state.yt_vid = vid
                st.session_state.yt_info = fetch_youtube_video_info(vid, api_key)
                
                with st.spinner("Fetching comments from YouTube API..."):
                    try:
                        comments = fetch_youtube_comments(vid, api_key, max_comments=max_comments)
                    except Exception as e:
                        st.error(f"Failed to fetch comments: {e}")
                        comments = []

                if not comments:
                    st.warning("No comments could be fetched. The video might have comments disabled or is private.")
                else:
                    st.success(f"Successfully fetched {len(comments)} comments. Running AI inference...")
                    
                    batch_size = 32
                    preds, confidences, texts, preprocessed_texts = [], [], [], []
                    progress_bar = st.progress(0)
                    total = len(comments)
                    
                    for i in range(0, total, batch_size):
                        batch = comments[i:i+batch_size]
                        pre_batch = [preprocess_text_full(c) for c in batch]
                        
                        try:
                            outs = nlp(pre_batch)
                        except Exception:
                            outs = [nlp(pb)[0] for pb in pre_batch]
                            
                        for out in outs:
                            scores = out if isinstance(out, list) else [out]
                            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                                scores = out
                            elif isinstance(out, dict):
                                scores = [out]
                                
                            try:
                                label, conf = get_top_prediction(scores)
                            except:
                                label, conf = get_top_prediction(outs)
                                
                            preds.append(normalize_label(label))
                            confidences.append(conf)
                            
                        texts.extend(batch)
                        preprocessed_texts.extend(pre_batch)
                        progress_bar.progress(min(1.0, (i+batch_size)/total))
                        
                    progress_bar.empty()

                    df_res = pd.DataFrame({
                        "Comment": texts,
                        "Preprocessed": preprocessed_texts,
                        "predicted_label": preds,
                        "Confidence": confidences
                    })

                    st.session_state.yt_df_res = df_res

    # --- Display YouTube Results if available in Session State ---
    if st.session_state.yt_df_res is not None:
        st.markdown("---")
        df_res = st.session_state.yt_df_res
        vid = st.session_state.yt_vid
        yt_info = st.session_state.yt_info

        # 1. Show Video Preview (if available)
        if yt_info:
            c1, c2 = st.columns([1, 4])
            with c1:
                st.image(yt_info["thumbnail"], use_container_width=True)
            with c2:
                st.markdown(f"#### {yt_info['title']}")
                st.caption(f"Channel: {yt_info['channel']} | Video ID: `{vid}`")

        # 2. Show Summary Metrics
        st.markdown("### 📊 Analysis Summary")
        counts = df_res["predicted_label"].value_counts()
        total_comments = len(df_res)
        
        c_net, c_tox, c_jud = st.columns(3)
        c_net.metric("✅ Neutral Comments", counts.get("Neutral", 0))
        c_tox.metric("🤬 Toxic Comments", counts.get("Toxic", 0))
        c_jud.metric("🎰 Online Gambling", counts.get("Online Gambling", 0))

        # 3. Charts & Word Cloud
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.markdown("#### Class Distribution")
            fig, ax = plt.subplots(figsize=(5,4))
            # Custom colors for pie chart
            colors_map = {"Neutral": "#2ecc71", "Toxic": "#e74c3c", "Online Gambling": "#e67e22"}
            colors = [colors_map.get(x, '#95a5a6') for x in counts.index]
            
            ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)
            
        with viz_col2:
            st.markdown("#### ☁️ Negative Word Cloud")
            st.caption("Common words in Toxic & Gambling comments")
            negative_text = " ".join(df_res[df_res["predicted_label"].isin(["Toxic", "Online Gambling"])]["Preprocessed"].tolist())
            
            if negative_text.strip():
                wordcloud = WordCloud(width=600, height=480, background_color='white', colormap='Reds').generate(negative_text)
                fig_wc, ax_wc = plt.subplots(figsize=(5,4))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No Toxic or Online Gambling comments found to generate a Word Cloud.")

        # 4. Filterable Interactive Table
        st.markdown("### 🔎 Interactive Results Table")
        
        # Multiselect for filtering
        selected_classes = st.multiselect(
            "Filter by Predicted Class:",
            options=["Neutral", "Toxic", "Online Gambling"],
            default=["Neutral", "Toxic", "Online Gambling"]
        )
        
        # Apply filter
        df_filtered = df_res[df_res["predicted_label"].isin(selected_classes)].copy()
        
        # Apply Color Styling to the Dataframe
        styled_df = df_filtered.style.apply(highlight_classes, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=350)

        # 5. Export / Download
        st.markdown("### 📥 Export Filtered Data")
        dl_col1, dl_col2 = st.columns([1, 3])
        with dl_col1:
            dl_format = st.selectbox("Select format:", ["CSV (.csv)", "Excel (.xlsx)"], label_visibility="collapsed")
        
        with dl_col2:
            if dl_format == "CSV (.csv)":
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="Download CSV File", 
                    data=csv, 
                    file_name=f"youtube_analysis_{vid}.csv", 
                    mime="text/csv",
                    type="primary"
                )
            else:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_filtered.to_excel(writer, index=False, sheet_name='Predictions')
                
                st.download_button(
                    label="Download Excel File",
                    data=buffer.getvalue(),
                    file_name=f"youtube_analysis_{vid}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

        if show_raw:
            st.markdown("#### All Raw Predictions")
            st.write(df_res)
