
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define preprocessing function for text
def preprocess_text(text):
    # Initialize WordNet Lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    # Remove non-alphabetic characters and tokenize
    text = ' '.join([word.lower() for word in nltk.word_tokenize(text) if word.isalpha()])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Lemmatize tokens
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

# Read data from CSV
data = pd.read_csv(r"C:\Users\MATANGEE\Downloads\jarvis-calling-hiring-contest\Resume\Resume.csv")

# Preprocess the 'resume_text' column
data['cleaned_resume'] = data['Resume_str'].apply(preprocess_text)

# Check class distribution
print(data['Category'].value_counts())

# Create TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_resume'])
y = data['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

# Train a RandomForestClassifier with class weights
clf = RandomForestClassifier(class_weight=class_weights_dict, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Preprocess text extracted from PDF
pdf_path = r"C:\Users\MATANGEE\Downloads\akshita-resume-4.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
pdf_text_cleaned = preprocess_text(pdf_text)
pdf_vector = vectorizer.transform([pdf_text_cleaned])

# Predict category for PDF
predicted_category = clf.predict(pdf_vector)[0]
print("Predicted category for PDF:", predicted_category)


# In[ ]:





# In[ ]:




