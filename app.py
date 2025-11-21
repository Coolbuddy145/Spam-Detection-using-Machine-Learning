import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- SETUP ---
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

# --- CUSTOM TEXT TRANSFORMATION ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in text]

    return " ".join(y)

# --- LOAD MODEL AND VECTORIZER ---
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🤖",
    layout="centered"
)

# --- CUSTOM CSS FOR DECORATION ---
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(120deg, #89f7fe, #66a6ff);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Title Styling */
    .title {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-top: 10px;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #f8f9fa;
        font-size: 1.1rem;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
    }

    /* Input Box */
    .stTextArea textarea {
        background-color: #ffffff;
        border: 3px solid #3b82f6;
        border-radius: 12px;
        color: #111827;
        font-size: 1.1rem;
        padding: 15px;
        box-shadow: 0 0 12px rgba(59,130,246,0.2);
        transition: 0.3s ease-in-out;
    }
    .stTextArea textarea:focus {
        border-color: #10b981;
        box-shadow: 0 0 15px rgba(16,185,129,0.4);
    }

    /* Button */
    div.stButton > button {
        background-color: #111827;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: scale(1.05);
    }

    /* Result Box */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-top: 25px;
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(10px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #f0f0f0;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown("<h1 class='title'>🤖 AI Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Classify messages as <b>Spam</b> or <b>Not Spam</b> using a trained NLP Model ⚡</p>", unsafe_allow_html=True)

# --- INPUT SECTION ---
input_sms = st.text_area("✉️ Enter your message below:")

# --- PREDICTION BUTTON ---
if st.button("🚀 Predict Now"):
    transformed_sms = transform_text(input_sms)

    if transformed_sms.strip() == "":
        st.warning("⚠️ Please enter a valid message before prediction.")
    else:
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0][result] * 100

        # --- RESULT DISPLAY ---
        if result == 1:
            st.markdown(f"""
                <div class='result-card' style='background: linear-gradient(90deg, #ff6a6a, #ff0000);'>
                    🚨 <b>SPAM MESSAGE DETECTED</b> 🚨 <br><br>
                    This message is likely spam.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-card' style='background: linear-gradient(90deg, #10b981, #34d399);'>
                    ✅ <b>SAFE MESSAGE</b> ✅ <br><br>
                    This message is not spam.</b>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

# --- FOOTER ---
st.markdown("<p class='footer'>Built by <b>Zaid Khan</b> | Powered by Machine Learning & NLTK 🌐</p>", unsafe_allow_html=True)
