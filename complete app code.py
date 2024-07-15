import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from PyPDF2 import PdfReader
import io
import spacy

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model for NER
import en_core_web_sm
nlp = en_core_web_sm.load()

# Define preprocessing function for text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([word.lower() for word in word_tokenize(text) if word.isalpha()])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Function to perform NER using spaCy
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to load data (now using st.cache_data)
@st.cache_data(persist=True)
def load_data():
    return pd.read_csv(r"C:\Users\MATANGEE\Downloads\jarvis-calling-hiring-contest\Resume\Resume.csv")

# Function to preprocess the data
@st.cache_data(persist=True)
def preprocess_data(data):
    data['cleaned_resume'] = data['Resume_str'].apply(preprocess_text)
    return data

# Function to train the model
@st.cache_data(persist=True)
def train_model(X, y):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
    clf = RandomForestClassifier(class_weight=class_weights_dict, random_state=42)
    clf.fit(X_train, y_train)
    return clf, vectorizer

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    try:
        with io.BytesIO(uploaded_file.read()) as f:
            pdf_reader = PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                pdf_text += pdf_reader.pages[page_num].extract_text()
    except Exception as e:
        st.error(f"An error occurred while extracting text: {e}")
    return pdf_text

# Simple chat function
def chat_with_bot(user_input, resume_text):
    response = f"You said: '{user_input}'. How can I assist you with this resume?"
    return response

# Define Streamlit app
def main():
    st.title('Resume Category Prediction and Chat')
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ('Home', 'Upload Resume', 'Chat'))

    if page == 'Home':
        st.write('## Home Page')
        st.write('Welcome to the Resume App!')

    elif page == 'Upload Resume':
        st.write('## Upload Resume')
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key='file_uploader')

        if uploaded_file is not None:
            if uploaded_file.size == 0:
                st.error("Uploaded file is empty. Please upload a valid PDF file.")
            else:
                st.write('### PDF Uploaded Successfully!')
                st.session_state['uploaded_file'] = uploaded_file

                pdf_text = extract_text_from_pdf(uploaded_file)

                # Display extracted text
                st.write('### Extracted Text')
                st.write(pdf_text)

                # Perform NER on extracted text
                st.write('### Named Entities (NER)')
                entities = perform_ner(pdf_text)
                for entity, label in entities:
                    st.write(f"{entity} ({label})")

                # Load and preprocess data
                data = load_data()
                data = preprocess_data(data)

                # Train model (if not cached)
                clf, vectorizer = train_model(data['cleaned_resume'], data['Category'])

                # Store model in session state for later use
                st.session_state['model'] = (clf, vectorizer)

                # Preprocess PDF text
                pdf_text_cleaned = preprocess_text(pdf_text)
                pdf_vector = vectorizer.transform([pdf_text_cleaned])

                # Predict category for PDF
                predicted_category = clf.predict(pdf_vector)[0]
                st.write('### Prediction Result')
                st.write(f'Predicted category for PDF: **{predicted_category}**')

    elif page == 'Chat':
        st.write('## Chat with the Bot')
        st.write('Start chatting about the uploaded resume!')

        uploaded_file = st.session_state.get('uploaded_file', None)
        if uploaded_file is not None:
            pdf_text = extract_text_from_pdf(uploaded_file)
            resume_text = preprocess_text(pdf_text)
            user_input = st.text_input('You:', '')
            if st.button('Send'):
                st.write('Bot:', chat_with_bot(user_input, resume_text))

# Run the app
if __name__ == '__main__':
    main()
