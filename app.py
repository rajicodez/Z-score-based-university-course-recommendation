import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

# --- 1. Data Loading, Preprocessing, and Model Training ---
# This entire section runs only once when the Flask application starts,
# so the data and model are ready to be used for all requests.

try:
    df = pd.read_csv('university_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'university_dataset.csv' not found. Please ensure it's in the correct location.")
    exit()

# Data Cleaning
df['Stream'] = df['Stream'].str.strip().str.lower()
df['District'] = df['District'].str.strip().str.lower()
df['Degree Program'] = df['Degree Program'].str.strip().str.lower()

# Get unique values for dropdowns
unique_streams = sorted(df['Stream'].unique().tolist())
unique_districts = sorted(df['District'].unique().tolist())

# One-Hot Encoding
encoder_stream = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_streams = encoder_stream.fit_transform(df[['Stream']])
encoded_stream_df = pd.DataFrame(encoded_streams, columns=encoder_stream.get_feature_names_out(['Stream']))

encoder_district = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_districts = encoder_district.fit_transform(df[['District']])
encoded_district_df = pd.DataFrame(encoded_districts, columns=encoder_district.get_feature_names_out(['District']))

# Create processed feature set
df_processed = pd.concat([df[['Min Z-score']], encoded_stream_df, encoded_district_df], axis=1)

# Create a mapping for program IDs
df['Program_ID'] = df['Degree Program'].astype('category').cat.codes
program_id_to_name = dict(enumerate(df['Degree Program'].astype('category').cat.categories))

# Define features (X) and target (y)
X = df_processed
y = df['Program_ID']

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# --- 2. Recommendation Functions ---
def recommend_courses_ml_based(z_score, stream, district, ml_model, encoder_stream, encoder_district, program_id_to_name, df_full_data):
    """
    Recommends university degree programs using a trained ML model.
    """
    stream = stream.strip().lower()
    district = district.strip().lower()

    # Create a DataFrame for the new student's input
    all_encoded_cols = encoder_stream.get_feature_names_out(['Stream']).tolist() + \
                       encoder_district.get_feature_names_out(['District']).tolist()
    student_input_df = pd.DataFrame(0, index=[0], columns=all_encoded_cols)
    student_input_df['Min Z-score'] = z_score

    # Set the appropriate one-hot encoded columns to 1
    stream_col = f'Stream_{stream}'
    if stream_col in student_input_df.columns:
        student_input_df[stream_col] = 1
    
    district_col = f'District_{district}'
    if district_col in student_input_df.columns:
        student_input_df[district_col] = 1

    # Reindex the input DataFrame to match the training data columns
    training_columns = df_processed.columns.tolist()
    student_input_df = student_input_df.reindex(columns=training_columns, fill_value=0)

    # Predict the program ID using the trained ML model
    predicted_program_id = ml_model.predict(student_input_df)[0]
    predicted_program_name = program_id_to_name.get(predicted_program_id, "Unknown Program")

    # Filter the original dataset to find courses matching the predicted program and criteria
    recommended_courses = df_full_data[
        (df_full_data['Stream'] == stream) &
        (df_full_data['District'] == district) &
        (df_full_data['Degree Program'] == predicted_program_name) &
        (df_full_data['Min Z-score'] <= z_score)
    ]

    if recommended_courses.empty:
        # Fallback: Recommend any course in the student's stream and district with sufficient Z-score
        recommended_courses_fallback = df_full_data[
            (df_full_data['Stream'] == stream) &
            (df_full_data['District'] == district) &
            (df_full_data['Min Z-score'] <= z_score)
        ].sort_values(by='Min Z-score', ascending=False)
        
        return recommended_courses_fallback[['Degree Program', 'Min Z-score']]

    recommended_courses = recommended_courses.sort_values(by='Min Z-score', ascending=False)
    return recommended_courses[['Degree Program', 'Min Z-score']]

# --- 3. Flask Application Routes ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        z_score = float(request.form['z_score'])
        stream = request.form['stream']
        district = request.form['district']

        ml_recommended_courses = recommend_courses_ml_based(
            z_score,
            stream,
            district,
            dt_classifier, 
            encoder_stream,
            encoder_district,
            program_id_to_name,
            df
        )
        
        if not ml_recommended_courses.empty:
            results_html = ml_recommended_courses.to_html(classes=['table-auto', 'w-full', 'text-left', 'bg-white', 'rounded-lg'])
        else:
            results_html = "<p class='text-gray-700'>No ML-based recommendations found for the given criteria.</p>"
        
        return render_template('index.html', 
                               results=results_html,
                               z_score=z_score,
                               stream=stream,
                               district=district,
                               unique_streams=unique_streams,
                               unique_districts=unique_districts)

    return render_template('index.html', 
                           results=None,
                           unique_streams=unique_streams,
                           unique_districts=unique_districts)

if __name__ == '__main__':
    app.run(debug=True)
