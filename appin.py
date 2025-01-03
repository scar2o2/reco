import pandas as pd
from flask import Flask, render_template, request
import pickle
import csv

app = Flask(__name__)

# Load datasets and model
try:
    crop_data = pd.read_csv('crop_data.csv')  # Ensure 'crop_data.csv' exists and is properly formatted
    techniques_data = pd.read_csv('techniquesf.csv')  # Ensure this CSV file exists
    model = pickle.load(open('crop_model.pkl', 'rb'))  # Ensure 'crop_model.pkl' exists
    df = pd.read_csv('crops.csv')  # Ensure 'crops.csv' exists and has the right format
    columns = df.drop(columns=["Recommended_Crop"]).columns.tolist()  # Assuming 'Recommended_Crop' column is in crops.csv
except Exception as e:
    print(f"Error loading datasets or model: {e}")
    crop_data, techniques_data, model, columns = None, None, None, []

# Function to read data from the CSV file and return as a list of dictionaries
def read_schemes_from_csv():
    schemes = []
    try:
        with open('schemes.csv', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                row['id'] = int(row['id'])  # Convert 'id' to integer
                schemes.append(row)
    except Exception as e:
        print(f"Error reading schemes.csv: {e}")
    return schemes

# Homepage route
@app.route('/')
def homepage():
    """Render the homepage."""
    return render_template('homepage.html')

# Crop recommendation page
@app.route('/chome')
def crop_recommendation():
    """Render the crop recommendation input page."""
    return render_template('index.html', columns=columns)

@app.route('/chome/recommend', methods=['POST'])
def recommend():
    """
    Process user input to recommend crops.
    """
    try:
        # Extract input data dynamically based on columns
        params = {col: float(request.form[col]) for col in columns}
        input_df = pd.DataFrame([params])
        probabilities = model.predict_proba(input_df)[0]
        recommended_crop = model.classes_[probabilities.argmax()]
        top_crops = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
        recommendations = [crop for crop, _ in top_crops]
        return render_template(
            'index.html',
            columns=columns,
            recommendation=recommended_crop,
            suggestions=recommendations
        )
    except Exception as e:
        return render_template(
            'index.html',
            columns=columns,
            error=f"An error occurred during recommendation: {str(e)}"
        )

# Display list of crops
@app.route('/crops')
def crops1():
    """Display a list of available crops."""
    crop_names = crop_data['Crop Name'].tolist()  # Ensure column name matches the CSV
    return render_template('crops.html', crop_names=crop_names)

# Display crop details
@app.route('/crops/<crop_name>')
def crop_detail(crop_name):
    """Display details for a specific crop."""
    try:
        crop_info = crop_data[crop_data['Crop Name'] == crop_name].iloc[0].to_dict()
        crop_names = crop_data['Crop Name'].tolist()
        return render_template('crop_detail.html', crop_name=crop_name, crop_info=crop_info, crop_names=crop_names)
    except IndexError:
        return f"Crop '{crop_name}' not found.", 404

# Agricultural techniques list
@app.route('/techniques')
def techniques():
    """Display a list of agricultural techniques."""
    technique_names = techniques_data['technique'].tolist() if techniques_data is not None else []
    return render_template('techniques1.html', technique_names=technique_names)

# Display technique details
@app.route('/techniques/<technique_name>')
def technique_detail(technique_name):
    """Display details for a specific agricultural technique."""
    try:
        technique_info = techniques_data[techniques_data['technique'] == technique_name].iloc[0].to_dict()
        technique_names = techniques_data['technique'].tolist()
        return render_template(
            'technique_detail1.html',
            technique_name=technique_name,
            technique_info=technique_info,
            technique_names=technique_names
        )
    except IndexError:
        return f"Technique '{technique_name}' not found.", 404

# Display list of schemes
@app.route('/sch')
def scheme_list():
    """Render the list of schemes."""
    schemes = read_schemes_from_csv()
    return render_template('schemes.html', schemes=schemes)

# Display scheme details
@app.route('/sch/scheme/<int:scheme_id>')
def scheme_detail(scheme_id):
    """Render details for a specific scheme."""
    schemes = read_schemes_from_csv()  # Load schemes dynamically from CSV
    # Find the scheme based on ID
    scheme_info = next((scheme for scheme in schemes if scheme['id'] == scheme_id), None)
    
    if scheme_info is None:
        return f"Scheme with ID {scheme_id} not found.", 404
    
    return render_template('scheme_details.html', scheme_info=scheme_info, schemes=schemes)

if __name__ == "__main__":
    app.run(debug=True)
