University Program Recommender
This is a Python-based web application designed to help students discover potential university degree programs based on their academic performance. The system uses a machine learning model to provide personalized recommendations.

Project Description
The University Program Recommender provides students with a user-friendly interface to input their G.C.E. Advanced Level (A/L) Z-score, a specific A/L stream (e.g., "physical science"), and their district. The application then processes this information to generate a list of eligible degree programs. It employs a dual-approach to recommendations: a rule-based system for general eligibility and a machine learning model for a more targeted prediction.

Features
Interactive Web Interface: A clean and simple web form built with Flask and styled with Tailwind CSS for an intuitive user experience.

Machine Learning-Powered Recommendations: Utilizes a Decision Tree Classifier from the scikit-learn library to predict the most probable degree program for a student.

Rule-Based Fallback: If the ML model's prediction doesn't have a perfect match in the dataset for a student's criteria, a fallback system recommends all available programs that meet the Z-score cutoff.

Automated Data Processing: Handles all data cleaning, one-hot encoding, and model training in the backend upon application startup.

Dynamic Dropdown Menus: The web form's dropdown menus for streams and districts are populated dynamically from the dataset, ensuring accurate and up-to-date options.

Getting Started
Follow these steps to set up and run the application locally.

Prerequisites
You must have Python installed. The required libraries can be installed using pip.

pip install Flask pandas scikit-learn

Installation and Setup
Clone or Download the Repository: Ensure you have all the project files in a single folder.

app.py (The Python Flask application)

university_dataset.csv (Your data file)

A subfolder named templates

Inside templates, an index.html file (The web frontend)

Your file structure should look exactly like this:

zscore/
├── app.py
├── university_dataset.csv
└── templates/
    └── index.html

Run the Application: Navigate to the project directory in your terminal or command prompt and execute the following command:

python app.py

Access the Web Interface: Once the server is running, open your web browser and go to http://127.0.0.1:5000/.

Technology Stack
Backend: Python, Flask

Data Science: Pandas, Scikit-learn

Frontend: HTML, Tailwind CSS
