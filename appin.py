import pandas as pd
from flask import Flask, render_template, request, jsonify
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

# Function to load crop data dynamically from advice.csv (from flask2)
def load_crop_data():
    crop_data = {}
    try:
        with open("advice.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                crop_data[row["Crop"].lower()] = {
                    "N_min": int(row["N_min"]),
                    "N_max": int(row["N_max"]),
                    "P_min": int(row["P_min"]),
                    "P_max": int(row["P_max"]),
                    "K_min": int(row["K_min"]),
                    "K_max": int(row["K_max"]),
                }
    except Exception as e:
        print(f"Error loading advice.csv: {e}")
    return crop_data

# Load the crop data when the app starts
CROP_DATABASE = load_crop_data()

# Function to read schemes from CSV and return as a list of dictionaries
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

# Function to read diseases from CSV
def read_csv():
    diseases = []
    try:
        with open('diseases.csv', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                diseases.append(row)
    except Exception as e:
        print(f"Error reading diseases.csv: {e}")
    return diseases

# Homepage route
@app.route('/')
def welc():
    """Render the homepage."""
    return render_template('welc.html')
@app.route('/home')
def homepage():
    """Render the homepage."""
    return render_template('homepageINT.html')

# Crop recommendation page
@app.route('/chome')
def crop_recommendation():
    """Render the crop recommendation input page."""
    return render_template('recommender.html', columns=columns)

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
    crop_names = crop_data['Crop Name'].tolist() if crop_data is not None else []
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

# Home page showing the list of diseases
@app.route('/dis')
def disease_list():
    """Display the list of diseases."""
    diseases = read_csv()
    return render_template('diseases.html', diseases=diseases)

# Dynamic route for each disease
@app.route('/dis/disease/<int:disease_id>')
def disease_detail(disease_id):
    """Display details for a specific disease."""
    diseases = read_csv()
    try:
        disease = diseases[disease_id]
        return render_template(
            'disease_details.html',
            disease=disease,
            diseases=diseases,
            current_disease=disease['Disease']
        )
    except IndexError:
        return f"Disease with ID {disease_id} not found.", 404

# Route to handle NPK calculations
@app.route('/npk')
def npk_input():
    """Render the NPK input page."""
    return render_template('advisor.html')

@app.route('/api/calculate', methods=['POST'])
def calculate_npk():
    data = request.json
    crop = data["crop"].lower()
    n, p, k = data["n"], data["p"], data["k"]
    acres = data["acres"]

    if crop not in CROP_DATABASE:
        return jsonify({
            "advice": [f"Crop '{crop}' is not in the database. Please choose a valid crop."]
        })

    crop_data = CROP_DATABASE[crop]
    advice = []

    # Check NPK values and calculate adjustments based on acreage
    if n < crop_data["N_min"]:
        amount = (crop_data["N_min"] - n) * acres
        advice.append(f"Add {amount} kg of Urea to increase Nitrogen for {acres} acre(s).")
        advice.append("Mix Urea evenly with the top 6-8 inches of soil and irrigate immediately to prevent nitrogen loss.")

    elif n > crop_data["N_max"]:
        amount = (n - crop_data["N_max"]) * acres
        advice.append(f"Reduce Nitrogen by {amount} kg for {acres} acre(s) using controlled-release fertilizers.")
        advice.append("Apply nitrogen in split doses during the crop's growth stages to avoid excessive buildup.")

    if p < crop_data["P_min"]:
        amount = (crop_data["P_min"] - p) * acres
        advice.append(f"Add {amount} kg of Superphosphate for Phosphorus for {acres} acre(s).")
        advice.append("Incorporate Superphosphate into the soil close to the root zone for better absorption.")

    elif p > crop_data["P_max"]:
        amount = (p - crop_data["P_max"]) * acres
        advice.append(f"Reduce Phosphorus by {amount} kg for {acres} acre(s) using phosphate binders.")
        advice.append("Avoid applying additional phosphorus fertilizers until soil levels stabilize.")

    if k < crop_data["K_min"]:
        amount = (crop_data["K_min"] - k) * acres
        advice.append(f"Use {amount} kg of Potash to increase Potassium for {acres} acre(s).")
        advice.append("Mix Potash thoroughly with the soil and water immediately for better uptake.")

    elif k > crop_data["K_max"]:
        amount = (k - crop_data["K_max"]) * acres
        advice.append(f"Reduce Potassium by {amount} kg for {acres} acre(s) using potassium-binding substances.")
        advice.append("Apply less potassium in subsequent crop cycles to balance the soil.")

    if not advice:
        advice.append(f"Your soil's NPK values are optimal for {acres} acre(s) of {crop} cultivation.")

    # Append minimum NPK requirements to the advice
    ideal_ranges = (
        f"Ideal NPK ranges for {crop.title()} are: "
        f"N: {crop_data['N_min']}-{crop_data['N_max']}, "
        f"P: {crop_data['P_min']}-{crop_data['P_max']}, "
        f"K: {crop_data['K_min']}-{crop_data['K_max']}."
    )
    advice.append(ideal_ranges)

    return jsonify({"advice": advice})

if __name__ == "__main__":
    app.run(debug=True)
