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
st.set_page_config(page_title="IndoBERT Classifier", layout="centered", initial_sidebar_state="expanded")

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
    '𝐚':'a', '𝐛':'b', '𝐜':'c', '𝐝':'d', '𝐞':'e', '𝐟':'f', '𝐠':'g', '𝐡':'h', '𝐢':'i', '𝐣':'j', '𝐤':'k', '𝐥':'l', '𝐦':'m', '𝐧':'n', '𝐨':'o', '𝐩':'p', '𝐪':'q', '𝐫':'r', '𝐬':'s', '𝐭':'t', '𝐮':'u', '𝐯':'v', '𝐰':'w', '𝐱':'x', '𝐲':'y', '𝐳':'z',
    '𝟎':'0', '𝟏':'1', '𝟐':'2', '𝟑':'3', '𝟒':'4', '𝟓':'5', '𝟔':'6', '𝟕':'7', '𝟖':'8', '𝟗':'9',

    # Sans-Serif Bold Italic
    '𝘼':'A', '𝘽':'B', '𝘾':'C', '𝘿':'D', '𝙀':'E', '𝙁':'F', '𝙂':'G', '𝙃':'H', '𝙄':'I', '𝙅':'J', '𝙆':'K', '𝙇':'L', '𝙈':'M', '𝙉':'N', '𝙊':'O', '𝙋':'P', '𝙌':'Q', '𝙍':'R', '𝙎':'S', '𝙏':'T', '𝙐':'U', '𝙑':'V', '𝙒':'W', '𝙓':'X', '𝙔':'Y', '𝙕':'Z',
    '𝙖':'a', '𝙗':'b', '𝙘':'c', '𝙙':'d', '𝙚':'e', '𝙛':'f', '𝙜':'g', '𝙝':'h', '𝙞':'i', '𝙟':'j', '𝙠':'k', '𝙡':'l', '𝙢':'m', '𝙣':'n', '𝙤':'o', '𝙥':'p', '𝙦':'q', '𝙧':'r', '𝙨':'s', '𝙩':'t', '𝙪':'u', '𝙫':'v', '𝙬':'w', '𝙭':'x', '𝙮':'y', '𝙯':'z',

    # Sans-Serif Bold
    '𝗔':'A', '𝗕':'B', '𝗖':'C', '𝗗':'D', '𝗘':'E', '𝗙':'F', '𝗚':'G', '𝗛':'H', '𝗜':'I', '𝗝':'J', '𝗞':'K', '𝗟':'L', '𝗠':'M', '𝗡':'N', '𝗢':'O', '𝗣':'P', '𝗤':'Q', '𝗥':'R', '𝗦':'S', '𝗧':'T', '𝗨':'U', '𝗩':'V', '𝗪':'W', '𝗫':'X', '𝗬':'Y', '𝗭':'Z',
    '𝗮':'a', '𝗯':'b', '𝗰':'c', '𝗱':'d', '𝗲':'e', '𝗳':'f', '𝗴':'g', '𝗵':'h', '𝗶':'i', '𝗷':'j', '𝗸':'k', '𝗹':'l', '𝗺':'m', '𝗻':'n', '𝗼':'o', '𝗽':'p', '𝗾':'q', '𝗿':'r', '𝘀':'s', '𝘁':'t', '𝘂':'u', '𝘃':'v', '𝘄':'w', '𝘅':'x', '𝘆':'y', '𝘇':'z',
    '𝟬':'0', '𝟭':'1', '𝟮':'2', '𝟯':'3', '𝟰':'4', '𝟱':'5', '𝟲':'6', '𝟳':'7', '𝟴':'8', '𝟵':'9',

    # Monospace (for 𝙿𝚛𝚘𝚋𝚎𝚝𝟾𝟻𝟻)
    '𝙰':'A', '𝙱':'B', '𝙲':'C', '𝙳':'D', '𝙴':'E', '𝗙':'F', '𝙶':'G', '𝙷':'H', '𝙸':'I', '𝙹':'J', '𝙺':'K', '𝙻':'L', '𝙼':'M', '𝙽':'N', '𝙾':'O', '𝙿':'P', '𝚀':'Q', '𝚁':'R', '𝚂':'S', '𝚃':'T', '𝚄':'U', '𝚅':'V', '𝚆':'W', '𝚇':'X', '𝚈':'Y', '𝚉':'Z',
    '𝚊':'a', '𝚋':'b', '𝚌':'c', '𝚍':'d', '𝚎':'e', '𝚏':'f', '𝚐':'g', '𝚑':'h', '𝚒':'i', '𝚓':'j', '𝚔':'k', '𝚕':'l', '𝚖':'m', '𝚗':'n', '𝚘':'o', '𝚙':'p', '𝚚':'q', '𝚛':'r', '𝚜':'s', '𝚝':'t', '𝚞':'u', '𝚟':'v', '𝚠':'w', '𝚡':'x', '𝚢':'y', '𝚣':'z',
    '𝟶':'0', '𝟷':'1', '𝟸':'2', '𝟹':'3', '𝟺':'4', '𝟻':'5', '𝟼':'6', '𝟽':'7', '𝟾':'8', '𝟿':'9',

    # Fraktur / Gothic (for 𝕻𝖚𝖑𝖆𝖚𝖜𝖎𝖓𝖟)
    '𝕬':'A', '𝕭':'B', '𝕮':'C', '𝕯':'D', '𝕰':'E', '𝕱':'F', '𝕲':'G', '𝕳':'H', '𝕴':'I', '𝕵':'J', '𝕶':'K', '𝕷':'L', '𝕸':'M', '𝕹':'N', '𝕺':'O', '𝕻':'P', '𝕼':'Q', '𝕽':'R', '𝕾':'S', '𝕿':'T', '𝖀':'U', '𝖁':'V', '𝖂':'W', '𝖃':'X', '𝖄':'Y', '𝖅':'Z',
    '𝖆':'a', '𝖇':'b', '𝖈':'c', '𝖉':'d', '𝖊':'e', '𝖋':'f', '𝖌':'g', '𝖍':'h', '𝖎':'i', '𝖏':'j', '𝖐':'k', '𝖑':'l', '𝖒':'m', '𝖓':'n', '𝖔':'o', '𝖕':'p', '𝖖':'q', '𝖗':'r', '𝖘':'s', '𝖙':'t', '𝖚':'u', '𝖛':'v', '𝖜':'w', '𝖝':'x', '𝖞':'y', '𝖟':'z',

    # Enclosed Alphanumerics (specifically for 🄿🅄🄻🄰🅄🅆🄸🄽)
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

# ---------------- Automatic processing for DataFrame 'all_data' if present ----------------
try:
    if 'all_data' in globals() and isinstance(all_data, pd.DataFrame) and 'text' in all_data.columns:
        print("Starting cleaning process with definitive normalization dictionary...")
        print("Step 1: Cleaning, font normalization, and URL removal...")
        all_data['clean_text'] = all_data['text'].apply(clean_text_modified)
        print("Step 2: Converting to lowercase...")
        all_data['clean_text'] = all_data['clean_text'].str.lower()
        print("Step 3: Removing stopwords...")
        if stop_remover is None and _SASTRAWI_AVAILABLE:
            try:
                stop_factory = StopWordRemoverFactory()
                stop_remover = stop_factory.create_stop_word_remover()
            except Exception:
                stop_remover = None
        if stop_remover is not None:
            all_data['clean_text'] = all_data['clean_text'].apply(lambda x: stop_remover.remove(x))
        else:
            print("Warning: Sastrawi stop_remover is not available; skipping stopword removal.")
        print("Step 4: Performing stemming...")
        if stemmer is None and _SASTRAWI_AVAILABLE:
            try:
                stem_factory = StemmerFactory()
                stemmer = stem_factory.create_stemmer()
            except Exception:
                stemmer = None
        if stemmer is not None:
            all_data['clean_text'] = all_data['clean_text'].apply(lambda x: safe_stemmer(x, stemmer))
        else:
            print("Warning: Sastrawi stemmer is not available; skipping stemming.")
        print("\nCleaning process completed.")
        print("\n--- Sample Results on 'all_data' ---")
        try:
            print(all_data[['text', 'clean_text']].head())
        except Exception:
            print("Cannot display sample results (environment might not support DataFrame printing).")
except Exception as e:
    print(f"Auto-processing all_data failed: {e}")

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

# ---------------- YouTube helpers ----------------
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

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

# ---------------- Sidebar UI ----------------
st.sidebar.header("Settings")
repo_input = st.sidebar.text_input("Model repo / folder", value=DEFAULT_REPO,
                                  help="Hugging Face repo (username/repo) or local path")
device_opt = st.sidebar.selectbox("Device", options=["auto", "cpu", "gpu"], index=0,
                                  help="auto -> use GPU if available")
show_raw = st.sidebar.checkbox("Show raw scores (for debugging)", value=False)
example_btn = st.sidebar.button("Use example text")

# ---------------- Main UI ----------------
st.title("🧪 IndoBERT — Text Classification")
st.subheader("Input: Single text or YouTube link (analyze comments)")

with st.spinner("Loading model (only once)..."):
    try:
        nlp, device_used = load_pipeline_hf(repo_input, device_choice=device_opt)
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Failed to load model from `{repo_input}`:\n{e}")
        st.code(tb)
        st.stop()

if device_used == 0:
    st.success("Model loaded — GPU will be used for inference.")
else:
    st.info("Model loaded — CPU will be used for inference.")

mode = st.radio("Select input mode:", ["Single text", "YouTube URL (comments)"])

if example_btn:
    default_text = "This product is very satisfying. Fast delivery and good quality."
else:
    default_text = ""

# ---------------- Single text mode ----------------
if mode == "Single text":
    text = st.text_area("Enter text to classify", value=default_text, height=140)
    if st.button("Predict single"):
        if not text or not text.strip():
            st.warning("Input cannot be empty.")
        else:
            with st.spinner("Performing preprocessing & inference..."):
                pre = preprocess_text_full(text)
                out = nlp(pre)
                
                if isinstance(out[0], list):
                    scores = out[0]
                else:
                    scores = out
                    
                top_label, top_score = get_top_prediction(scores)
                display_label = normalize_label(top_label)

            st.markdown("### 🔎 Final Prediction")
            st.write("**Original:**", text)
            st.write("**Preprocessed:**", pre)
            st.metric(label="Predicted class", value=f"{display_label}", delta=f"{top_score:.4f}")
            st.caption("The probability in the metric is for the selected class.")

            df_data = []
            for x in scores:
                df_data.append({
                    "label": normalize_label(x["label"]), 
                    "score": x["score"]
                })
            
            df = pd.DataFrame(df_data)

            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            st.markdown("#### Class Probabilities")
            st.bar_chart(df.set_index("label"))

            if show_raw:
                st.markdown("#### Raw scores")
                st.json(scores)

# ---------------- YouTube comments mode ----------------
else:
    youtube_url = st.text_input("Enter YouTube link (or video ID directly):", value="")
    max_comments = st.slider("Maximum number of comments", min_value=10, max_value=1000, value=200, step=10)
    analyze_btn = st.button("Analyze comments")

    if analyze_btn:
        vid = extract_video_id(youtube_url)
        if not vid:
            st.error("Cannot extract video ID. Ensure the URL is correct.")
        else:
            api_key = None
            try:
                api_key = st.secrets["YOUTUBE_API_KEY"]
            except Exception:
                api_key = os.environ.get("YOUTUBE_API_KEY")
                
            if not api_key:
                st.error("YouTube API key not found. Set `YOUTUBE_API_KEY` in Streamlit secrets or env var.")
            else:
                with st.spinner("Fetching comments from YouTube..."):
                    try:
                        comments = fetch_youtube_comments(vid, api_key, max_comments=max_comments)
                    except Exception as e:
                        st.error(f"Failed to fetch comments: {e}")
                        comments = []

                if not comments:
                    st.warning("No comments fetched (or comments are disabled).")
                else:
                    st.success(f"Fetched {len(comments)} comments — running preprocessing & inference...")
                    batch_size = 32
                    preds = []
                    confidences = []
                    texts = []
                    preprocessed_texts = []
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
                            if isinstance(out, list):
                                scores = out
                            else:
                                scores = [out] if not isinstance(out, list) and not isinstance(out, dict) else (out if isinstance(out, list) else [out])
                                
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
                        "comment": texts,
                        "preprocessed": preprocessed_texts,
                        "predicted_label": preds,
                        "confidence": confidences
                    })

                    counts = df_res["predicted_label"].value_counts()

                    st.markdown("### 📊 Comment Class Distribution")
                    fig, ax = plt.subplots()

                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                    st.markdown("### 🔎 Results Table")
                    st.dataframe(df_res.head(200))

                    st.markdown("### 📥 Download Results")
                    
                    # Dropdown form untuk format file download
                    dl_format = st.selectbox("Select download format:", ["CSV (.csv)", "Excel (.xlsx)"])
                    
                    if dl_format == "CSV (.csv)":
                        csv = df_res.to_csv(index=False)
                        st.download_button(
                            label="Download File", 
                            data=csv, 
                            file_name=f"yt_comments_pred_{vid}.csv", 
                            mime="text/csv"
                        )
                    else:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df_res.to_excel(writer, index=False, sheet_name='Predictions')
                        
                        st.download_button(
                            label="Download File",
                            data=buffer.getvalue(),
                            file_name=f"yt_comments_pred_{vid}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    if show_raw:
                        st.markdown("#### All predictions")
                        st.write(df_res)

# Footer / notes
st.markdown("---")
st.caption("Developed with ❤️ by Group 4 - Deep Learning - 2025")
