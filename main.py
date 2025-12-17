import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import openai
from openai import AsyncOpenAI
import faiss
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import os
import io
from io import BytesIO
import zipfile
import chardet
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
import traceback
import requests
import asyncio
import json
import dropbox
from dropbox.files import WriteMode
import base64
import zipfile
import gspread
from googleapiclient.discovery import build
from google.oauth2 import service_account
from collections import deque
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logging.basicConfig(level=logging.INFO)

# ---------------------------
# GLOBAL VARS
# ---------------------------
desc_sheet_id = st.secrets['DESC_GSHEET_ID']
anagrafica_sheet_id = st.secrets['ANAGRAFICA_GSHEET_ID']
LANG_NAMES = {
    "IT": "italiano",
    "EN": "inglese",
    "FR": "francese",
    "DE": "tedesco",
    "ES": "spagnolo"
}
LANG_LABELS = {v.capitalize(): k for k, v in LANG_NAMES.items()}


# ---------------------------
# 🔐 Setup API keys and credentials
# ---------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

def check_openai_key():
    try:
        openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True, None
    except Exception as e:  
        msg = str(e).lower()
        return False, msg


# =========================================================
# ⚙️ CONFIGURAZIONE FILE DB + GITHUB
# =========================================================

GITHUB_REPO = "MarcoRipari/Gestione-Ecom"
GITHUB_PATH = "data/translations_db.json"
GITHUB_BRANCH = "main"

# =========================================================
# 🌐 FUNZIONI DI GESTIONE SU GITHUB
# =========================================================

def download_translation_db_from_github():
    """Scarica il file JSON delle traduzioni da GitHub e lo restituisce come oggetto Python"""
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("⚠️ Nessun GITHUB_TOKEN trovato tra i secrets.")
        return []

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    headers = {"Authorization": f"token {github_token}"}

    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            if "content" in data:
                content = base64.b64decode(data["content"]).decode("utf-8")
                print("✅ DB traduzioni caricato da GitHub.")
                return json.loads(content)
            else:
                print("⚠️ Nessun contenuto trovato nel file GitHub.")
                return []
        elif r.status_code == 404:
            print("⚠️ File delle traduzioni non trovato su GitHub. Creerò un nuovo DB.")
            return []
        else:
            print(f"⚠️ Errore scaricando DB da GitHub: {r.status_code} - {r.text}")
            return []
    except Exception as e:
        print(f"❌ Errore durante il download del DB: {e}")
        return []


def upload_translation_db_to_github(db, original_db_json):
    """Carica o aggiorna il file delle traduzioni su GitHub solo se ci sono modifiche"""
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("⚠️ Nessun GITHUB_TOKEN trovato tra i secrets. Upload annullato.")
        return

    # 🔍 Confronto con il contenuto originale
    new_db_json = json.dumps(db, ensure_ascii=False, indent=2)
    if new_db_json == original_db_json:
        print("ℹ️ Nessuna nuova traduzione aggiunta: nessun upload necessario.")
        return  # Non aggiorna GitHub se identico

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    headers = {"Authorization": f"token {github_token}"}

    try:
        content = base64.b64encode(new_db_json.encode("utf-8")).decode("utf-8")

        # Ottieni SHA del file esistente (necessario per aggiornamento)
        sha = None
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            sha = r.json().get("sha")

        message = "Aggiornamento automatico del DB traduzioni"
        data = {
            "message": message,
            "content": content,
            "branch": GITHUB_BRANCH,
        }
        if sha:
            data["sha"] = sha  # necessario se il file esiste già

        r = requests.put(url, headers=headers, json=data)

        if r.status_code in (200, 201):
            print("✅ File delle traduzioni aggiornato su GitHub!")
        else:
            print(f"❌ Errore aggiornando su GitHub: {r.status_code} - {r.text}")

    except Exception as e:
        print(f"❌ Errore durante l'upload su GitHub: {e}")


# =========================================================
# 🧩 FUNZIONI DI GESTIONE DEL DB (IN MEMORIA)
# =========================================================

def find_translation(db, text_it, target_lang):
    """Cerca una traduzione esistente nel DB"""
    text_it = str(text_it).strip().lower()
    for entry in db:
        if entry.get("it", "").strip().lower() == text_it:
            return entry.get(target_lang)
    return None


def add_translation(db, text_it, lang, translated_text):
    """Aggiunge o aggiorna una traduzione nel DB (solo in memoria)"""
    text_it = str(text_it).strip()
    for entry in db:
        if entry.get("it", "").strip().lower() == text_it.lower():
            entry[lang] = translated_text
            break
    else:
        db.append({"it": text_it, lang: translated_text})


# =========================================================
# 🧠 FUNZIONI DI TRADUZIONE
# =========================================================

def create_translator(source, target):
    return GoogleTranslator(source=source, target=target)


def safe_translate(text, translator, db):
    """Traduci testo con gestione errori e uso del DB GitHub"""
    time.sleep(0.1)
    try:
        if not text or str(text).strip() == "":
            return ""

        text_it = str(text).strip()
        target_lang = translator.target

        # 1️⃣ Controlla se esiste già nel DB
        cached = find_translation(db, text_it, target_lang)
        if cached:
            return cached

        # 2️⃣ Se non esiste → traduci e aggiungi
        translated = translator.translate(text_it)
        add_translation(db, text_it, target_lang, translated)
        return translated

    except Exception as e:
        print(f"❌ Errore durante la traduzione: {e}")
        return str(text)


def translate_column_parallel(col_values, source, target, db, max_workers=5):
    """Traduci una colonna mantenendo l'ordine originale"""
    translator = create_translator(source, target)
    results = [None] * len(col_values)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_translate, text, translator, db): i for i, text in enumerate(col_values)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Errore riga {idx}: {e}")
                results[idx] = str(col_values[idx])

    return results


# ---------------------------
# 📊 Google Sheets
# ---------------------------
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
)
gsheet_client = gspread.authorize(credentials)
drive_service = build('drive', 'v3', credentials=credentials)

def get_sheet(sheet_id, tab):
    spreadsheet = gsheet_client.open_by_key(sheet_id)
    worksheets = spreadsheet.worksheets()
    
    # Confronto case-insensitive per maggiore robustezza
    for ws in worksheets:
        if ws.title.strip().lower() == tab.strip().lower():
            return ws

    # Se non trovato, lo crea
    return spreadsheet.add_worksheet(title=tab, rows="10000", cols="50")

def append_to_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    df = df.fillna("").astype(str)
    values = df.values.tolist()
    sheet.append_rows(values, value_input_option="RAW")  # ✅ chiamata unica

def append_logs(sheet_id, logs_data):
    sheet = get_sheet(sheet_id, "logs")
    values = [list(log.values()) for log in logs]
    sheet.append_rows(values, value_input_option="RAW")
    
def append_log(sheet_id, log_data):
    sheet = get_sheet(sheet_id, "logs")
    sheet.append_row(list(log_data.values()), value_input_option="RAW")


# ---------------------------
# 📦 Embedding & FAISS Setup
# ---------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=st.secrets["HF_TOKEN"])
    return model.to("cpu")

model = load_model()

def embed_texts(texts: List[str], batch_size=32) -> List[List[float]]:
    return model.encode(texts, show_progress_bar=False, batch_size=batch_size).tolist()

def hash_dataframe_and_weights(df: pd.DataFrame, col_weights: Dict[str, float]) -> str:
    df_bytes = pickle.dumps((df.fillna("").astype(str), col_weights))
    return hashlib.md5(df_bytes).hexdigest()

def build_faiss_index(df: pd.DataFrame, col_weights: Dict[str, float], cache_dir="faiss_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hash_dataframe_and_weights(df, col_weights)
    cache_path = os.path.join(cache_dir, f"{cache_key}.index")

    if os.path.exists(cache_path):
        index = faiss.read_index(cache_path)
        return index, df

    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                weight = col_weights.get(col, 1)
                if weight > 0:
                    parts.append((f"{col}: {row[col]} ") * int(weight))
        texts.append(" ".join(parts))

    vectors = embed_texts(texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, cache_path)

    return index, df

def retrieve_similar(query_row: pd.Series, df: pd.DataFrame, index, k=5, col_weights: Dict[str, float] = {}):
    parts = []
    for col in df.columns:
        if pd.notna(query_row[col]):
            weight = col_weights.get(col, 1)
            if weight > 0:
                parts.append((f"{col}: {query_row[col]} ") * int(weight))
    query_text = " ".join(parts)

    query_vector = embed_texts([query_text])[0]
    D, I = index.search(np.array([query_vector]).astype("float32"), k)

    # 🔍 DEBUG
    logging.info(f"QUERY TEXT: {query_text[:300]} ...")
    logging.info(f"INDICI trovati: {I[0]}")
    logging.info(f"Distanze: {D[0]}")
    
    return df.iloc[I[0]]

def estimate_embedding_time(df: pd.DataFrame, col_weights: Dict[str, float], sample_size: int = 10) -> float:
    """
    Stima il tempo totale per embeddare tutti i testi del dataframe.
    """
    texts = []
    for _, row in df.head(sample_size).iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                weight = col_weights.get(col, 1)
                if weight > 0:
                    parts.append((f"{col}: {row[col]} ") * int(weight))
        texts.append(" ".join(parts))

    start = time.time()
    _ = embed_texts(texts)
    elapsed = time.time() - start
    avg_time_per_row = elapsed / sample_size
    total_estimated_time = avg_time_per_row * len(df)

    return total_estimated_time

def benchmark_faiss(df, col_weights, query_sample_size=10):
    
    st.markdown("### ⏱️ Benchmark FAISS + Embedding")

    start_embed = time.time()
    texts = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" * int(col_weights.get(col, 1))
                 for col in df.columns if pd.notna(row[col])]
        texts.append(" ".join(parts))
    vectors = embed_texts(texts)
    embed_time = time.time() - start_embed

    start_faiss = time.time()
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, "tmp_benchmark.index")
    index_time = time.time() - start_faiss

    index_size = os.path.getsize("tmp_benchmark.index")

    # Test query
    query_times = []
    for i in range(min(query_sample_size, len(df))):
        qtext = texts[i]
        start_q = time.time()
        _ = index.search(np.array([vectors[i]]).astype("float32"), 5)
        query_times.append(time.time() - start_q)

    avg_query_time = sum(query_times) / len(query_times)

    st.write({
        "🚀 Tempo embedding totale (s)": round(embed_time, 2),
        "📄 Tempo medio per riga (ms)": round(embed_time / len(df) * 1000, 2),
        "🏗️ Tempo costruzione FAISS (s)": round(index_time, 2),
        "💾 Dimensione index (KB)": round(index_size / 1024, 1),
        "🔍 Tempo medio query (ms)": round(avg_query_time * 1000, 2),
    })

    os.remove("tmp_benchmark.index")


# ---------------------------
# 🧠 Prompting e Generazione
# ---------------------------
def build_unified_prompt(row, col_display_names, selected_langs, simili=None):
    # Costruzione scheda tecnica
    fields = []
    for col in col_display_names:
        if col in row and pd.notna(row[col]):
            label = col_display_names[col]
            fields.append(f"- {label}: {row[col]}")
    product_info = "\n".join(fields)

    # Elenco lingue in stringa
    lang_list = ", ".join([LANG_NAMES.get(lang, lang) for lang in selected_langs])

    # Descrizioni simili
    sim_text = ""
    if simili is not None and not simili.empty:
        sim_lines = []
        for _, ex in simili.iterrows():
            dl = ex.get("Description", "").strip()
            db = ex.get("Description2", "").strip()
            if dl and db:
                sim_lines.append(f"- {dl}\n  {db}")
        if sim_lines:
            sim_text = "\nDescrizioni simili:\n" + "\n".join(sim_lines)

    #incipit_seeds = ["Descrittivo", "Pratico", "Poetico"]
    incipit_seeds = ["SEO-oriented", "Descrittivo", "Pratico", "Classico", "Informativo", "Accattivante"]
    # Prompt finale
    prompt = f"""Scrivi due descrizioni coerente con le INFO ARTICOLO per una calzatura da vendere online (e-commerce) in ciascuna delle seguenti lingue: {lang_list}.

>>> GUIDA STILE E LINGUAGGIO
- Stile di apertura: {random.choice(incipit_seeds)}
- Tono: {", ".join(selected_tones)}
- Lingua: adatta al paese target
- Mai usare: Codice, Nome, Marca, Colore
- Non usare descrizioni di colore, finitura, effetto estetico o trattamento visivo (es. usato, effetto usato, scolorito, vintage, lavato, distressed)
- Descrivi solo materiali e componenti strutturali, NON finiture o trattamenti estetici
- Utilizza esclusivamente il tipo di calzatura passato nelle info articoli
- Non usare generi o età (es. maschile/femminile, bambino/bambina)
- Evita le percentuali materiali
- Evita la durezza del materiale (soffice come, morbidezza, sensazione al tatto, ecc...)
- Evita qualsiasi linguaggio sensoriale
- Evita frasi sulla facilità d'uso generico, non possiamo garantirlo.
- Evita verbi che implicano promesse o benefici soggettivi (es. garantire, offrire, assicurare, migliorare, accompagnare il piede)
- Evita ripetizioni ravvicinate di parole o strutture sintattiche
- Se lo stesso materiale o elemento compare più volte, raggruppa le informazioni in un’unica frase
- Alterna la struttura delle frasi per evitare sequenze ripetitive (es. "in pelle", "anch’essa in pelle")
- NON descrivere il comfort come sensazione o beneficio percepito
- Verifica la correttezza della descrizione rispetto alla stagione tra le info articolo
- Non usare **alcuna formattazione Markdown** nell'output

>>> PAROLE DA EVITARE (anche implicite)
- velcro → usa "strappo"
- velluto → usa "velour" o "suede"
- primavera, estate, autunno, inverno (e derivati)

>>> ESEMPI DI ERRORI DA EVITARE
❌ "velluto" → ✅ "velour" o "suede"
❌ "primaverile" → ✅ descrizione neutra sulla stagione
❌ "per bambina" → ✅ descrizione neutra
❌ "first shoes" → ✅ scarpe
❌ "prime scarpe" → ✅ scarpe
❌ "scarpa da primi passi" → ✅ scarpe
❌ "in pelle con effetto usato" → ✅ "in pelle"

>>> ESEMPIO DI STILE CORRETTO
❌ "Progettate per garantire comfort e ammortizzazione durante l’uso quotidiano."
✅ "La suola presenta una struttura multistrato e uno spessore adatto all’uso quotidiano."

>>> REGOLE
- desc_lunga: {desc_lunga_length} parole → enfasi su caratteristiche costruttive, materiali e struttura della calzatura
- desc_breve: {desc_breve_length} parole → adatta a social media o schede prodotto rapide
- Output JSON: {{"it":{{"desc_lunga":"...","desc_breve":"..."}}, "en":{{...}}, "fr":{{...}}, "de":{{...}}}}

>>> INFO ARTICOLO
{product_info}

{sim_text}
Dopo aver generato le descrizioni, rileggi e correggi eventuali errori grammaticali o di genere **prima** di produrre l'output finale JSON.

>>> CONTROLLO FINALE
Controlla attentamente che le descrizioni:
- rispettino tutte le regole fornite (parole vietate, formato, tono, ecc.)
- non contengano errori grammaticali, di concordanza o di traduzione in nessuna lingua
- in italiano, controlla sempre il genere e il numero dei sostantivi (es. "questi sandali", non "queste sandali")
- se trovi errori di grammatica, rigenera o correggi la frase **prima di fornire l'output finale**
- se una frase non descrive una caratteristica fisica, costruttiva o verificabile della calzatura, rigenera o correggi la frase **prima di fornire l'output finale**
- fornisci l'output finale **solo dopo** aver verificato che sia grammaticalmente e stilisticamente corretto in tutte le lingue
"""
    return prompt


client = AsyncOpenAI(api_key=openai.api_key)


async def async_generate_description(prompt: str, idx: int, use_model: str):
    temperature = random.uniform(0.9, 1.2)
    presence_penalty = random.uniform(0.4, 0.8)
    
    if len(prompt) < 50:
        return idx, {
            "result": prompt,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0}
        }
        
    try:
        response = await client.chat.completions.create(
            model=use_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=0.95,
            frequency_penalty=0.4,
            presence_penalty=presence_penalty,
            max_tokens=3000
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        data = json.loads(content)
        return idx, {"result": data, "usage": usage.model_dump()}
    except Exception as e:
        return idx, {"error": str(e)}


async def generate_all_prompts(prompts: list[str], model: str) -> dict:
    tasks = [async_generate_description(prompt, idx, model) for idx, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return dict(results)


def calcola_tokens(df_input, col_display_names, selected_langs, selected_tones, desc_lunga_length, desc_breve_length, k_simili, faiss_index, DEBUG=False):
    if df_input.empty:
        return None, None, "❌ Il CSV è vuoto"

    row = df_input.iloc[0]

    simili = pd.DataFrame([])
    if k_simili > 0 and faiss_index:
        index, index_df = faiss_index
        simili = retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)

    prompt = build_unified_prompt(
        row=row,
        col_display_names=col_display_names,
        selected_langs=selected_langs,
        simili=simili
    )

    # Token estimation (~4 chars per token)
    num_chars = len(prompt)
    token_est = num_chars // 4
    cost_est = round(token_est / 1000 * 0.001, 6)

    st.code(prompt)
    st.markdown(f"📊 **Prompt Length**: {num_chars} caratteri ≈ {token_est} token")
    st.markdown(f"💸 **Costo stimato per riga**: ${cost_est:.6f}")

    return token_est, cost_est, prompt


# ---------------------------
# DropBox
# ---------------------------

def get_dropbox_access_token():
    refresh_token = st.secrets["DROPBOX_REFRESH_TOKEN"]
    client_id = st.secrets["DROPBOX_CLIENT_ID"]
    client_secret = st.secrets["DROPBOX_CLIENT_SECRET"]

    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    response = requests.post(
        "https://api.dropbox.com/oauth2/token",
        headers={
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]

def get_dropbox_client():
    access_token = get_dropbox_access_token()
    return dropbox.Dropbox(access_token)
    
def upload_to_dropbox(dbx, folder_path: str, file_name: str, file_bytes: bytes):
    dbx_path = f"{folder_path}/{file_name}"
    try:
        dbx.files_create_folder_v2(folder_path)
    except dropbox.exceptions.ApiError:
        pass  # cartella già esiste
    try:
        dbx.files_upload(file_bytes, dbx_path, mode=WriteMode("overwrite"))
        
        st.success(f"✅ File caricato su Dropbox: {dbx_path}")
    except Exception as e:
        st.error(f"❌ Errore upload su Dropbox: {e}")
        
def upload_csv_to_dropbox(dbx, folder_path: str, file_name: str, file_bytes: bytes):
    dbx_path = f"{folder_path}/{file_name}"
    try:
        dbx.files_create_folder_v2(folder_path)
    except dropbox.exceptions.ApiError:
        pass  # cartella già esiste
    try:
        dbx.files_upload(file_bytes, dbx_path, mode=WriteMode("overwrite"))
        
        st.success(f"✅ CSV caricato su Dropbox: {dbx_path}")
    except Exception as e:
        st.error(f"❌ Errore caricando CSV su Dropbox: {e}")

def download_csv_from_dropbox(dbx, folder_path: str, file_name: str) -> io.BytesIO:
    file_path = f"{folder_path}/{file_name}"

    try:
        metadata, res = dbx.files_download(file_path)
        return io.BytesIO(res.content), metadata
    except dropbox.exceptions.ApiError as e:
        # Se l'errore è 'path/not_found' -> file mancante
        if (hasattr(e.error, "is_path") and e.error.is_path() 
                and e.error.get_path().is_not_found()):
            return None, None
        else:
            # altri errori (permessi, connessione, ecc.)
            st.error(f"Errore scaricando da Dropbox: {e}")
            return None, None
            
def format_dropbox_date(dt):
    if dt is None:
        return "Data non disponibile"

    # Dropbox restituisce sempre datetime tz-aware in UTC, ma nel dubbio gestiamo anche i naïve
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    # Convertiamo in fuso orario italiano
    dt_italy = dt.astimezone(ZoneInfo("Europe/Rome"))

    # Data odierna in Italia
    oggi = datetime.now(ZoneInfo("Europe/Rome")).date()

    mesi_it = [
        "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
        "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
    ]

    if dt_italy.date() == oggi:
        return f"Oggi alle {dt_italy.strftime('%H:%M')}"
    else:
        mese = mesi_it[dt_italy.month - 1]
        return f"{dt_italy.day:02d} {mese} {dt_italy.year} - {dt_italy.strftime('%H:%M')}"


# ---------------------------
# Funzioni varie
# ---------------------------
def read_csv_auto_encoding(uploaded_file, separatore=None):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    uploaded_file.seek(0)  # Rewind after read
    if separatore:
        return pd.read_csv(uploaded_file, sep=separatore, encoding=encoding, dtype=str)
    else:
        return pd.read_csv(uploaded_file, encoding=encoding, dtype=str)

def not_in_array(array, list):
    missing = not all(col in array for col in list)
    return missing

# ---------------------------
# 📦 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Gestione ECOM", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# ---------------------------
# ☰ SIDEBAR
# ---------------------------
with st.sidebar:
    #DEBUG = st.checkbox("🪛 Debug")
    DEBUG = True
    # Togliere per riattivare password e nome
    if DEBUG:
        st.session_state.user = {
            "data": "data",
            "email": "test@test.it",
            "nome": "GUEST",
            "cognome": "Test2",
            "username": "Username",
            "role": "admin"
        }

    if "user" not in st.session_state or st.session_state.user is None:
        page = "Home"
        st.markdown("## 🔑 Login")
        with st.form("login_user"):
            email = st.text_input("Username")
            password = st.text_input("Password", type="password")

            login_button = st.form_submit_button("Accedi")
            
        if login_button:
            if login(email, password):
                st.rerun()  # ricarica subito la pagina senza messaggio
    else:
        user = st.session_state.user
        st.write(f"Accesso eseguito come: {user["nome"]}")

        menu_item_list = [{"name":"Home", "icon":"house", "role":["guest","logistica","customer care","admin"]},
                          {"name":"Descrizioni", "icon":"list", "role":["customer care","admin"]},
                         ]
        
        submenu_item_list = [{"main":"Catalogo", "name":"Trova articolo", "icon":"search", "role":["logistica","customer care","admin"]},
                             {"main":"Catalogo", "name":"Aggiungi ordini stagione", "icon":"plus", "role":["logistica","customer care","admin"]},
                             {"main":"Ordini", "name":"Dashboard", "icon":"bar-chart", "role":["admin"]},
                             {"main":"Ordini", "name":"Importa", "icon":"plus", "role":["admin"]},
                             {"main":"Foto", "name":"Gestione", "icon":"gear", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Riscatta SKU", "icon":"repeat", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Aggiungi SKUs", "icon":"plus", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Storico", "icon":"book", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Aggiungi prelevate", "icon":"hand-index", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Giacenze", "name":"Importa", "icon":"download", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Giacenze", "name":"Per corridoio", "icon":"1-circle", "role":["guest","logistica","admin"]},
                             {"main":"Giacenze", "name":"Per corridoio/marchio", "icon":"2-circle", "role":["guest","logistica","admin"]},
                             {"main":"Giacenze", "name":"Aggiorna anagrafica", "icon":"refresh", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Giacenze", "name":"Old import", "icon":"download", "role":["admin"]},
                             {"main":"Ferie", "name":"Report", "icon":"list", "role":["admin"]},
                             {"main":"Ferie", "name":"Aggiungi ferie", "icon":"plus", "role":["admin"]},
                             {"main":"Admin", "name":"Aggiungi utente", "icon":"plus", "role":["admin"]}
                            ]
        
        menu_items = []
        icon_items = []
        for item in menu_item_list:
            if user["role"] in item["role"]:
                menu_items.append(item["name"])
                icon_items.append(item["icon"])
        
        
        st.markdown("## 📋 Menu")
        # --- Menu principale verticale ---
        main_page = option_menu(
            menu_title=None,
            options=menu_items,
            icons=icon_items,
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "2px",
                    "padding": "5px 10px",
                    "border-radius": "5px",
                    "--hover-color": "#e0e0e0",
                },
                "nav-link-selected": {
                    "background-color": "#4CAF50",
                    "color": "white",
                    "border-radius": "5px",
                },
            },
        )

        # Rimuovo icone/emoji per gestire page name
        main_page_name = main_page

        page = main_page_name  # default

        submenu_items = []
        submenu_icons = []
        for item in submenu_item_list:
            if main_page == item["main"] and user["role"] in item["role"]:
                submenu_items.append(item["name"])
                submenu_icons.append(item["icon"])
                
        if submenu_items:
            sub_page = option_menu(
                menu_title=None,
                options=submenu_items,
                icons=submenu_icons,
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                    "nav-link": {
                        "font-size": "15px",
                        "text-align": "left",
                        "margin": "2px",
                        "padding": "5px 15px",
                        "border-radius": "5px",
                        "--hover-color": "#e0e0e0",
                    },
                    "nav-link-selected": {
                        "background-color": "#4CAF50",
                        "color": "white",
                        "border-radius": "5px",
                    },
                },
            )
            page = f"{main_page_name} - {sub_page}"


if page == "Descrizioni":
    st.header("📥 Caricamento CSV dei prodotti")
    
    uploaded = st.file_uploader("Carica un file CSV", type="csv")
    
    if uploaded:
        df_input = read_csv_auto_encoding(uploaded)
        st.session_state["df_input"] = df_input
         # ✅ Inizializza variabili di stato se non esistono
        if "col_weights" not in st.session_state:
            st.session_state.col_weights = {}
        if "col_display_names" not in st.session_state:
            st.session_state.col_display_names = {}
        if "selected_cols" not in st.session_state:
            st.session_state.selected_cols = []
        if "config_ready" not in st.session_state:
            st.session_state.config_ready = False
        if "generate" not in st.session_state:
            st.session_state.generate = False
        st.success("✅ File caricato con successo!")

    # 📊 Anteprima dati
    if "df_input" in st.session_state:
        df_input = st.session_state.df_input
        st.subheader("🧾 Anteprima CSV")
        st.dataframe(df_input.head())

        # 🧩 Configurazione colonne
        with st.expander("⚙️ Configura colonne per il prompt", expanded=True):
            st.markdown("### 1. Seleziona colonne")
            available_cols = [col for col in df_input.columns if col not in ["Description", "Description2"]]
    
            def_column = ["Saison", "Silouhette",
                          "sole_material_zalando",
                          "shoe_fastener_zalando",
                          "upper_material_zalando",
                          "futter_zalando",
                          "Sp.feature"
                         ]
            trans_def_colum = {"Saison": "Stagione",
                               "Silouhette": "Tipo di calzatura",
                               "sole_material_zalando": "Soletta interna",
                               "shoe_fastener_zalando": "Chiusura",
                               "upper_material_zalando": "Tomaia",
                               "futter_zalando": "Fodera interna",
                               "Sp.feature": "Caratteristica"
                              }
            def_col_weights = {"Saison": 4,
                               "Silouhette": 5,
                               "sole_material_zalando": 3,
                               "shoe_fastener_zalando": 1,
                               "upper_material_zalando": 3,
                               "futter_zalando": 3,
                               "Sp.feature": 1
                              }
    
            missing = not_in_array(df_input.columns, def_column)
            if missing:
                def_column = []
                
            st.session_state.selected_cols = st.multiselect("Colonne da includere nel prompt", options=available_cols, default=def_column)
    
            if st.session_state.selected_cols:
                if st.button("▶️ Procedi alla configurazione colonne"):
                    st.session_state.config_ready = True
    
            if st.session_state.get("config_ready"):
                st.markdown("### 2. Configura pesi ed etichette")
                for col in st.session_state.selected_cols:
                    st.session_state.col_weights.setdefault(col, def_col_weights[col])
                    st.session_state.col_display_names.setdefault(col, col)
    
                    cols = st.columns([2, 3])
                    with cols[0]:
                        st.session_state.col_weights[col] = st.slider(
                            f"Peso: {col}", 0, 5, st.session_state.col_weights[col], key=f"peso_{col}"
                        )
                    with cols[1]:
                        st.session_state.col_display_names[col] = st.text_input(
                            #f"Etichetta: {col}", value=st.session_state.col_display_names[col], key=f"label_{col}"
                            f"Etichetta: {col}", value=trans_def_colum[col], key=f"label_{col}"
                        )
    
        # 🌍 Lingue e parametri
        with st.expander("🌍 Selezione Lingue & Parametri"):
            settings_col1, settings_col2, settings_col3 = st.columns(3)
            with settings_col1:
                marchio = st.radio(
                    "Seleziona il marchio",
                    ["NAT", "FAL", "VB", "FM", "WZ", "CC"],
                    horizontal = False
                )
                use_simili = st.checkbox("Usa descrizioni simili (RAG)", value=True)
                k_simili = 2 if use_simili else 0

                use_model = st.radio("Seleziona modello GPT", ["gpt-4o-mini", "gpt-4o"], index=0, horizontal = True)
    
            with settings_col2:
                selected_labels = st.multiselect(
                    "Lingue di output",
                    options=list(LANG_LABELS.keys()),
                    default=["Italiano", "Inglese", "Francese", "Tedesco", "Spagnolo"]
                )
                selected_langs = [LANG_LABELS[label] for label in selected_labels]
                
                selected_tones = st.multiselect(
                    "Tono desiderato",
                    ["informale", "conversazionale", "chiaro e diretto", "professionale", "amichevole", "accattivante", "descrittivo", "tecnico", "ironico", "minimal", "user friendly", "SEO-friendly", "SEO-optimized"],
                    default=["informale", "conversazionale", "chiaro e diretto", "user friendly", "SEO-friendly", "SEO-optimized"]
                )
    
            with settings_col3:
                desc_lunga_length = st.selectbox("Lunghezza descrizione lunga", ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"], index=5)
                desc_breve_length = st.selectbox("Lunghezza descrizione breve", ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"], index=1)
    
        # 💵 Stima costi
        if st.button("💰 Stima costi generazione"):
            token_est, cost_est, prompt = calcola_tokens(
                df_input=df_input,
                col_display_names=st.session_state.col_display_names,
                selected_langs=selected_langs,
                selected_tones=selected_tones,
                desc_lunga_length=desc_lunga_length,
                desc_breve_length=desc_breve_length,
                k_simili=k_simili,
                faiss_index=st.session_state.get("faiss_index"),
                DEBUG=True
            )
            if token_est:
                st.info(f"""
                📊 Token totali: ~{token_est}
                💸 Costo stimato: ${cost_est:.6f}
                """)
    
        # 🪄 Generazione descrizioni
        openai_check, openai_check_msg = check_openai_key()
        if not openai_check:
            st.error("❌ La chiave OpenAI non è valida o mancante. Inserisci una chiave valida prima di generare descrizioni.")
            st.error(openai_check_msg)
        else:
            if st.button("🚀 Genera Descrizioni"):
                st.session_state["generate"] = True
            
            if st.session_state.get("generate"):
                logs = []
                try:
                    with st.spinner("📚 Carico storico e indice FAISS..."):
                        tab_storico = f"STORICO_{marchio}"
                        data_sheet = get_sheet(desc_sheet_id, tab_storico)
                        df_storico = pd.DataFrame(data_sheet.get_all_records()).tail(500)
            
                        if "faiss_index" not in st.session_state:
                            index, index_df = build_faiss_index(df_storico, st.session_state.col_weights)
                            st.session_state["faiss_index"] = (index, index_df)
                        else:
                            index, index_df = st.session_state["faiss_index"]
            
                    # ✅ Recupera descrizioni già esistenti su GSheet
                    st.info("🔄 Verifico se alcune righe sono già state generate...")
                    existing_data = {}
                    already_generated = {lang: [] for lang in selected_langs}
                    rows_to_generate = []
            
                    for lang in selected_langs:
                        try:
                            tab_df = pd.DataFrame(get_sheet(desc_sheet_id, lang).get_all_records())
                            tab_df = tab_df[["SKU", "Description", "Description2"]].dropna(subset=["SKU"])
                            tab_df["SKU"] = tab_df["SKU"].astype(str)
                            existing_data[lang] = tab_df.set_index("SKU")
                        except:
                            existing_data[lang] = pd.DataFrame(columns=["Description", "Description2"])

                    unique_sku_prefixes = {}
                    for i, row in df_input.iterrows():
                        sku = str(row.get("SKU", "")).strip()
                        if not sku:
                            rows_to_generate.append(i)
                            continue
                    
                        all_present = True
                        for lang in selected_langs:
                            df_lang = existing_data.get(lang)
                            if df_lang is None or sku not in df_lang.index:
                                all_present = False
                                break
                            desc = df_lang.loc[sku]
                            if not desc["Description"] or not desc["Description2"]:
                                all_present = False
                                break
                    
                        if all_present:
                            # ✅ SKU già presente in tutti i fogli
                            for lang in selected_langs:
                                desc = existing_data[lang].loc[sku]
                                output_row = row.to_dict()
                                output_row["Description"] = desc["Description"]
                                output_row["Description2"] = desc["Description2"]
                                already_generated[lang].append(output_row)
                        else:
                            prefix = sku[:13]
                    
                            # 🔍 Cerca se esiste già una SKU con questo prefisso in existing_data
                            found_existing = False
                            for lang in selected_langs:
                                df_lang = existing_data.get(lang)
                                if df_lang is not None:
                                    # Controlla se esiste uno SKU con lo stesso prefisso
                                    match = [s for s in df_lang.index if s.startswith(prefix)]
                                    if match:
                                        desc = df_lang.loc[match[0]]
                                        output_row = row.to_dict()
                                        output_row["Description"] = desc["Description"]
                                        output_row["Description2"] = desc["Description2"]
                                        already_generated[lang].append(output_row)
                                        found_existing = True
                    
                            # Se nessuna SKU con quel prefisso è già presente → generala ora
                            if not found_existing:
                                if prefix not in unique_sku_prefixes:
                                    unique_sku_prefixes[prefix] = i
                                    rows_to_generate.append(i)
            
                    df_input_to_generate = df_input.iloc[rows_to_generate]
            
                    # Costruzione dei prompt
                    all_prompts = []
                    with st.spinner("✍️ Costruisco i prompt..."):
                        for _, row in df_input_to_generate.iterrows():
                            simili = retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights) if k_simili > 0 else pd.DataFrame([])
                            
                            prompt = build_unified_prompt(row, st.session_state.col_display_names, selected_langs, simili=simili)
                            all_prompts.append(prompt)
            
                    with st.spinner("🚀 Generazione asincrona in corso..."):
                        if use_model == "mistral-medium":
                            results = asyncio.run(generate_all_prompts_mistral(all_prompts, use_model))
                        elif use_model == "deepseek-chimera":
                            results = asyncio.run(generate_all_prompts_deepseek(all_prompts, use_model))
                        else:
                            results = asyncio.run(generate_all_prompts(all_prompts, use_model))
                    
                    # Parsing risultati
                    all_outputs = already_generated.copy()
                    prefix_to_output = {lang: {} for lang in selected_langs}
                    
                    for i, (_, row) in enumerate(df_input_to_generate.iterrows()):
                        result = results.get(i, {})
                        sku = str(row.get("SKU", "")).strip()
                        prefix = sku[:13]
                        if "error" in result:
                            logs.append({
                                "utente": st.session_state.user["username"],
                                "sku": row.get("SKU", ""),
                                "status": f"Errore: {result['error']}",
                                "prompt": all_prompts[i],
                                "output": "",
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            continue

                        sku_generate_lista = []
                        for lang in selected_langs:
                            output_row = row.to_dict()
                            lang_data = result.get("result", {}).get(lang.lower(), {})
                            descr_lunga = lang_data.get("desc_lunga", "").strip()
                            descr_breve = lang_data.get("desc_breve", "").strip()
                            output_row["Description"] = descr_lunga
                            output_row["Description2"] = descr_breve
                            all_outputs[lang].append(output_row)
                            prefix_to_output[lang][prefix] = output_row
            
                        log_entry = {
                            "utente": st.session_state.user["username"],
                            "sku": row.get("SKU", ""),
                            "status": "OK",
                            "prompt": all_prompts[i],
                            "output": json.dumps(result["result"], ensure_ascii=False),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        if "usage" in result:
                            usage = result["usage"]
                            log_entry.update({
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                                "estimated_cost_usd": round(usage.get("total_tokens", 0) / 1000 * 0.001, 6)
                            })
                        logs.append(log_entry)
                    for i, row in df_input.iterrows():
                        sku = str(row.get("SKU", "")).strip()
                        prefix = sku[:13]
                        if prefix in prefix_to_output[selected_langs[0]] and i not in rows_to_generate:
                            for lang in selected_langs:
                                copied_row = prefix_to_output[lang][prefix].copy()
                                new_row = row.copy()
                                #copied_row["SKU"] = sku  # sostituisci con lo SKU corrente
                                new_row["Description"] = copied_row.get("Description", "")
                                new_row["Description2"] = copied_row.get("Description2", "")
                                all_outputs[lang].append(new_row)
                                #all_outputs[lang].append(copied_row)
                                

                    # 🔄 Salvataggio solo dei nuovi risultati
                    with st.spinner("📤 Salvataggio nuovi dati..."):
                        try:
                            for lang in selected_langs:
                                df_out = pd.DataFrame(all_outputs[lang])
                                #df_new = df_out[df_out["SKU"].isin(df_input_to_generate["SKU"].astype(str))]
                                # Recupera gli SKU già presenti nello sheet
                                try:
                                    sheet_df = pd.DataFrame(get_sheet(desc_sheet_id, lang).get_all_records())
                                    sheet_df["SKU"] = sheet_df["SKU"].astype(str)
                                    existing_skus = set(sheet_df["SKU"].tolist())
                                except:
                                    existing_skus = set()

                                df_new = df_out[~df_out["SKU"].astype(str).isin(existing_skus)]
        
                                if not df_new.empty:
                                    append_to_sheet(desc_sheet_id, lang, df_new)

                            append_logs(desc_sheet_id, logs)
                        except Exception as e:
                            st.warning(f"Errore: {e}")

                    
                    # 📦 ZIP finale
                    with st.spinner("📦 Generazione ZIP..."):
                        translation_db = download_translation_db_from_github()
                        original_db_json = json.dumps(translation_db, ensure_ascii=False, indent=2)
                        
                        mem_zip = BytesIO()
                        with zipfile.ZipFile(mem_zip, "w") as zf:
                            for lang in selected_langs:
                                df_out = pd.DataFrame(all_outputs[lang])
                                df_out["Code langue"] = lang.lower()
                                df_out['Subtitle_trad'] = translate_column_parallel(df_out['Subtitle'].fillna("").tolist(),source='it', target=lang.lower(), db=translation_db, max_workers=5)
                                df_out['Subtile2_trad'] = translate_column_parallel(df_out['Subtile2'].fillna("").tolist(),source='it', target=lang.lower(), db=translation_db, max_workers=5)

                                df_export = pd.DataFrame({
                                    "skucolore": df_out.get("skucolore", ""),
                                    f"Modello ({lang.lower()})": df_out.get("Short_title", ""),
                                    f"Variante ({lang.lower()})": df_out.get("Subtitle_trad", ""),
                                    f"Colore ({lang.lower()})": df_out.get("Subtile2_trad", ""),
                                    f"Descrizione ({lang.lower()})": df_out.get("Description", ""),
                                    f"Descrizione 2 ({lang.lower()})": df_out.get("Description2", "")
                                })
                                zf.writestr(f"descrizioni_{lang}.csv", df_export.to_csv(index=False).encode("utf-8"))
                        mem_zip.seek(0)

                        # Aggiorno il file della traduzioni
                        upload_translation_db_to_github(translation_db, original_db_json)

                        now = datetime.now(ZoneInfo("Europe/Rome"))
                        file_name = f"descrizioni_{now.strftime('%d-%m-%Y_%H-%M-%S')}.zip"
                        # Carico il file su dropbox
                        try:
                            file_bytes = mem_zip.getvalue()
                            folder_path = "/CATALOGO/DESCRIZIONI"  # cartella su Dropbox
                            access_token = get_dropbox_access_token()
                            dbx = dropbox.Dropbox(access_token)
                            upload_to_dropbox(dbx, folder_path, file_name, file_bytes)
                        except Exception as e:
                            st.error(f"❌ Errore durante l'upload su Dropbox: {e}")
                            
                    st.success("✅ Tutto fatto!")
                    st.download_button("📥 Scarica descrizioni (ZIP)", mem_zip, file_name=file_name)
                    st.session_state["generate"] = False
            
                except Exception as e:
                    st.error(f"Errore durante la generazione: {str(e)}")
                    st.text(traceback.format_exc())
    
        # 🔍 Prompt Preview & Benchmark
        with st.expander("🔍 Strumenti di debug & Anteprima"):
            row_index = st.number_input("Indice riga per anteprima", 0, len(df_input) - 1, 0)
            test_row = df_input.iloc[row_index]
    
            if st.button("💬 Mostra Prompt di Anteprima"):
                with st.spinner("Generazione..."):
                    try:
                        if desc_sheet_id:
                            tab_storico = f"STORICO_{marchio}"
                            data_sheet = get_sheet(desc_sheet_id, tab_storico)
                            df_storico = pd.DataFrame(data_sheet.get_all_records()).tail(500)
                            if "faiss_index" not in st.session_state:
                                index, index_df = build_faiss_index(df_storico, st.session_state.col_weights)
                                st.session_state["faiss_index"] = (index, index_df)
                            else:
                                index, index_df = st.session_state["faiss_index"]
                            simili = (
                                retrieve_similar(test_row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)
                                if k_simili > 0 else pd.DataFrame([])
                            )
                        else:
                            simili = pd.DataFrame([])
    
                        image_url = test_row.get("Image 1", "")

                        prompt_preview = build_unified_prompt(test_row, st.session_state.col_display_names, selected_langs, simili=simili)
                        st.expander("📄 Prompt generato").code(prompt_preview, language="markdown")
                    except Exception as e:
                        st.error(f"Errore: {str(e)}")
    
            if st.button("🧪 Esegui Benchmark FAISS"):
                with st.spinner("In corso..."):
                    benchmark_faiss(df_input, st.session_state.col_weights)
