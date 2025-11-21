import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------------------------------
# 🔥 FIX: LOAD TOKENIZER + STOPWORDS WITHOUT DOWNLOAD
# -----------------------------------------------------
# Streamlit Cloud cannot download nltk data, so we use:
#   - nltk.tokenize.ToktokTokenizer (built-in, no download)
#   - manually defined STOPWORDS list

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()

# Custom stopwords list to avoid nltk download
STOPWORDS = set(stopwords.words('english')) if stopwords.words else {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now"
}

ps = PorterStemmer()

# -----------------------------------------------------
# TEXT PREPROCESSING (NO NLTK DOWNLOAD NEEDED)
# -----------------------------------------------------
def transform_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)  # no download needed

    filtered = []
    for w in tokens:
        if w.isalnum():
            filtered.append(w)

    filtered = [w for w in filtered if w not in STOPWORDS]
    stemmed = [ps.stem(w) for w in filtered]

    return " ".join(stemmed)


# -----------------------------------------------------
# LOAD YOUR MODEL + TFIDF
# -----------------------------------------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# -----------------------------------------------------
# STREAMLIT PAGE UI
# -----------------------------------------------------
st.set_page_config(page_title="AI Spam Detector", page_icon="🤖")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #89f7fe, #66a6ff);
        font-family: 'Poppins', sans-serif;
    }
    .title {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        color: #eef2ff;
        margin-bottom: 30px;
    }
    .result-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 20px;
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(10px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🤖 AI Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Classify messages as <b>Spam</b> or <b>Not Spam</b> using NLP ⚡</p>", unsafe_allow_html=True)

input_sms = st.text_area("✉️ Enter your message below:")

if st.button("🚀 Predict Now"):
    processed = transform_text(input_sms)

    if processed.strip() == "":
        st.warning("⚠ Please enter a valid message.")
    else:
        vector = tfidf.transform([processed])
        result = model.predict(vector)[0]

        if result == 1:
            st.markdown("""
                <div class='result-card' style='background: linear-gradient(90deg, #ff4b4b, #ff0000);'>
                    🚨 <b>SPAM MESSAGE DETECTED</b>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-card' style='background: linear-gradient(90deg, #00c851, #007e33);'>
                    ✅ <b>SAFE MESSAGE</b>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

st.markdown("<p style='text-align:center;color:white;'>Built by <b>Zaid Khan</b> | Machine Learning & NLP 🔥</p>", unsafe_allow_html=True)
