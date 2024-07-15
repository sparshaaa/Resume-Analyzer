# Resume-Analyzer
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Category Prediction and Chat Application</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        ul li::before {
            content: '•';
            color: #007BFF;
            display: inline-block;
            width: 1em;
            margin-left: -1em;
        }
    </style>
</head>
<body>

<h1>Resume Category Prediction and Chat Application</h1>

<p>Welcome to the Resume Category Prediction and Chat Application! This Streamlit-based web app allows users to upload resumes in PDF format, extract text from them, perform Named Entity Recognition (NER) to identify important entities, and predict the resume category using a pre-trained machine learning model. Additionally, users can chat with the bot to discuss their resumes.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#code-explanation">Code Explanation</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#notes">Notes</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>This project provides a streamlined way to analyze resumes and predict their categories. It leverages machine learning, natural language processing, and interactive chat functionalities to offer a comprehensive tool for resume analysis.</p>

<h2 id="features">Features</h2>
<ul>
    <li><strong>Home Page</strong>: A welcome page with introductory information.</li>
    <li><strong>Upload Resume</strong>: Allows users to upload a PDF resume, extract text, perform NER, and predict the resume category.</li>
    <li><strong>Chat</strong>: An interactive chat interface to discuss the uploaded resume with the bot.</li>
</ul>

<h2 id="installation">Installation</h2>
<h3>Prerequisites</h3>
<ul>
    <li>Python 3.7 or higher</li>
    <li>pip (Python package installer)</li>
</ul>

<h3>Steps</h3>
<pre>
<code>
1. Clone the repository:
    git clone https://github.com/yourusername/resume-category-prediction.git
    cd resume-category-prediction

2. Install the required packages:
    pip install -r requirements.txt
</code>
</pre>

<h2 id="usage">Usage</h2>
<pre>
<code>
1. Navigate to the project directory:
    cd resume-category-prediction

2. Run the Streamlit app:
    streamlit run app.py

3. Open the provided URL in your web browser to access the application.
</code>
</pre>

<h2 id="project-structure">Project Structure</h2>
<pre>
<code>
resume-category-prediction/
├── app.py                   # Main Streamlit application file
├── requirements.txt         # Required Python packages
├── README.md                # Project README file
└── data/
    └── Resume.csv           # Dataset containing resume data
</code>
</pre>

<h2 id="code-explanation">Code Explanation</h2>
<h3>Main Application (<code>app.py</code>)</h3>
<ul>
    <li><strong>Imports and Downloads</strong>: Import necessary libraries and download required NLTK data.</li>
    <li><strong>Preprocessing Functions</strong>:
        <ul>
            <li><code>preprocess_text(text)</code>: Tokenizes, removes stopwords, and lemmatizes the text.</li>
            <li><code>perform_ner(text)</code>: Uses spaCy to perform Named Entity Recognition.</li>
        </ul>
    </li>
    <li><strong>Data Handling Functions</strong>:
        <ul>
            <li><code>load_data()</code>: Loads the resume dataset (cached for efficiency).</li>
            <li><code>preprocess_data(data)</code>: Preprocesses the resume data.</li>
            <li><code>train_model(X, y)</code>: Trains the machine learning model (cached for efficiency).</li>
            <li><code>extract_text_from_pdf(uploaded_file)</code>: Extracts text from the uploaded PDF resume.</li>
        </ul>
    </li>
    <li><strong>Chat Function</strong>:
        <ul>
            <li><code>chat_with_bot(user_input, resume_text)</code>: Simple chat function for user interaction.</li>
        </ul>
    </li>
    <li><strong>Streamlit App Definition</strong>:
        <ul>
            <li><code>main()</code>: Defines the main structure of the Streamlit app, handling navigation between pages.</li>
        </ul>
    </li>
</ul>

<h2 id="data">Data</h2>
<p>The application uses a CSV file (<code>Resume.csv</code>) containing resume data for training the model. Ensure the CSV file is placed in the specified path (<code>data/Resume.csv</code>).</p>

<h2 id="notes">Notes</h2>
<ul>
    <li>Make sure all necessary NLTK data is downloaded.</li>
    <li>Customize paths and model parameters as needed.</li>
    <li>Use the provided <code>requirements.txt</code> file to install the necessary packages.</li>
</ul>

<h2 id="contributing">Contributing</h2>
<p>Contributions are welcome! Please follow these steps:</p>
<pre>
<code>
1. Fork the repository.
2. Create a new branch (<code>git checkout -b feature/your-feature</code>).
3. Commit your changes (<code>git commit -m 'Add some feature'</code>).
4. Push to the branch (<code>git push origin feature/your-feature</code>).
5. Create a new Pull Request.
</code>
</pre>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for more details.</p>

</body>
</html>