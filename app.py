from flask import Flask, jsonify, request
from flask_cors import CORS  # Import Flask-CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
app = Flask(__name__)
CORS(app)
# Load your trained models here
insertion_contenu_model_drupal = joblib.load('insertion_contenu_model_drupal.pkl')
ingenieur_systeme_model_drupal = joblib.load('ingenieur_systeme_model_drupal.pkl')
ingenieur_test_model_drupal = joblib.load('ingenieur_test_model_drupal.pkl')
seo_model_drupal = joblib.load('seo_model_drupal.pkl')
integration_model_drupal = joblib.load('integration_model_drupal.pkl')
infographie_model_drupal = joblib.load('infographie_model_drupal.pkl')
formation_model_drupal = joblib.load('formation_model_drupal.pkl')
cost_model=joblib.load('cost_model.pkl')
encoder = joblib.load('encoder.pkl')

# Define routes for predictions
def preprocess_input(data):
    # Ensure the input data is in a DataFrame format
    df = pd.DataFrame(data)

    # Encode 'Ressources' column using the loaded encoder
    resources_encoded = encoder.transform(df[['Ressources']])
    resources_encoded_df = pd.DataFrame(resources_encoded, columns=encoder.get_feature_names_out(['Ressources']))

    # Concatenate encoded features with the original DataFrame
    df_encoded = pd.concat([df.drop(columns=['Ressources']), resources_encoded_df], axis=1)
    
    # Define the features
    X = df_encoded[['J/H Vendus', 'Coût unitaire'] + list(df_encoded.columns[df_encoded.columns.str.startswith('Ressources_')])]
    return X

@app.route('/predict/cost', methods=['POST'])
def predict_cost():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Preprocess the input data
        X = preprocess_input(data)
        
        # Make predictions
        predictions = cost_model.predict(X)

        formatted_predictions = [round(pred) for pred in predictions]
        # Return the predictions as a JSON response
        return jsonify({'predictions': formatted_predictions})
    except Exception as e:
        return jsonify({'error': str(e)})
        
@app.route('/predict/insertion_contenu', methods=['POST'])
def predict_insertion_contenu():
    try:
        content = request.json
        analyste_concepteur = float(content['Analyste concepteur'])
        gestion_coordination = float(content['Gestion et coordination du projet'])

        prediction = insertion_contenu_model_drupal.predict([[analyste_concepteur, gestion_coordination]])[0]
        prediction = round(prediction, 2)
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/ingenieur_systeme', methods=['POST'])
def predict_ingenieur_systeme():
    try:
        content = request.json
        gestion_coordination = float(content['Gestion et coordination du projet'])

        if gestion_coordination < 5:
            ingenieur_systeme_pred = 0.5
        elif gestion_coordination < 6:
            ingenieur_systeme_pred = 1
        else:
            ingenieur_systeme_pred = 2

        return jsonify({'prediction': ingenieur_systeme_pred}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/ingenieur_test', methods=['POST'])
def predict_ingenieur_test():
    try:
        content = request.json
        ingenieur_systeme_pred = float(content['Ingénieur système'])
        gestion_coordination = float(content['Gestion et coordination du projet'])

        if ingenieur_systeme_pred <= 0.5:
            ingenieur_test_pred = 2
        else:
            ingenieur_test_pred = ingenieur_test_model_drupal.predict([[ingenieur_systeme_pred, gestion_coordination]])[0]
            ingenieur_test_pred = round(ingenieur_test_pred, 2)
        return jsonify({'prediction': ingenieur_test_pred}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/seo', methods=['POST'])
def predict_seo():
    try:
        content = request.json
        analyste_concepteur = float(content['Analyste concepteur'])
        gestion_coordination = float(content['Gestion et coordination du projet'])

        prediction = seo_model_drupal.predict([[analyste_concepteur, gestion_coordination]])[0]
        rounded_prediction = round(prediction, 2)
        return jsonify({'prediction': rounded_prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/integration', methods=['POST'])
def predict_integration():
    try:
        content = request.json
        ingenieur_test = float(content['Ingénieur test'])
        ingenieur_systeme = float(content['Ingénieur système'])
        infographie = float(content['Infographie '])

        prediction = integration_model_drupal.predict([[ingenieur_test, ingenieur_systeme, infographie]])[0]
        rounded_prediction = round(prediction, 2)
        return jsonify({'prediction': rounded_prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/infographie', methods=['POST'])
def predict_infographie():
    try:
        content = request.json
        analyste_concepteur = float(content['Analyste concepteur'])
        insertion_contenu = float(content['Insertion contenu'])

        prediction = infographie_model_drupal.predict([[analyste_concepteur, insertion_contenu]])[0]
        rounded_prediction = round(prediction, 2)
        return jsonify({'prediction': rounded_prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/formation', methods=['POST'])
def predict_formation():
    try:
        content=request.json
        gestion_coordination = float(content['Gestion et coordination du projet'])
        
        if gestion_coordination < 5:
            formation_pred = 0.5
        elif gestion_coordination < 6:
            formation_pred = 1
        else:
            formation_pred = 2
        return jsonify({'prediction': formation_pred}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/all', methods=['POST'])
def predict_all():
    try:
        content = request.json  # Get JSON data from POST request
        analyste_concepteur = float(content['Analyste concepteur'])
        gestion_coordination = float(content['Gestion et coordination du projet'])

        # Create data frames with the appropriate feature names
        df_insertion_contenu = pd.DataFrame([[analyste_concepteur, gestion_coordination]], columns=['Analyste concepteur', 'Gestion et coordination du projet'])
        
       
        df_seo = pd.DataFrame([[analyste_concepteur, gestion_coordination]], columns=['Analyste concepteur', 'Gestion et coordination du projet'])
     
        df_formation = pd.DataFrame([[gestion_coordination]], columns=['Gestion et coordination du projet'])

        # Predict Insertion Contenu
        prediction_IC = insertion_contenu_model_drupal.predict([[analyste_concepteur, gestion_coordination]])[0]

        # Predict Ingénieur Système verified
        if gestion_coordination < 5:
            prediction_IS = 0.5
        elif gestion_coordination < 6:
            prediction_IS = 1
        else:
            prediction_IS = 2

        # Predict Ingénieur Test verified
        if prediction_IS <= 0.5:
            prediction_IT = 2
        else:
            df_ingenieur_test = pd.DataFrame([[prediction_IS, gestion_coordination]], columns=['Ingénieur système', 'Gestion et coordination du projet'])
            prediction_IT = ingenieur_test_model_drupal.predict([[prediction_IS, gestion_coordination]])[0]

        # Predict SEO verified
        prediction_SEO = seo_model_drupal.predict([[analyste_concepteur, gestion_coordination]])[0]

        df_infographie = pd.DataFrame([[analyste_concepteur, prediction_IC]], columns=['Analyste concepteur', 'Insertion contenu'])

        # Predict Infographie verified
        prediction_Infographie = infographie_model_drupal.predict([[analyste_concepteur, prediction_IC]])[0]

        # Predict Integration verified
        df_integration = pd.DataFrame([[prediction_IT, prediction_IS, prediction_Infographie]], columns=['Ingénieur test', 'Ingénieur système', 'Infographie '])
        prediction_Integration = integration_model_drupal.predict([[prediction_IT, prediction_IS, prediction_Infographie]])[0]

        # Predict Formation
        if gestion_coordination < 5:
            formation_pred = 0.5
        elif gestion_coordination < 6:
            formation_pred = 1
        else:
            formation_pred = 2

        gestion_coordination = round(gestion_coordination, 2)
        analyste_concepteur = round(analyste_concepteur, 2)
        prediction_Infographie = round(prediction_Infographie, 2)
        prediction_Integration = round(prediction_Integration, 2)
        prediction_IC = round(prediction_IC, 2)
        prediction_IT = round(prediction_IT, 2)
        prediction_IS = round(prediction_IS, 2)
        prediction_SEO = round(prediction_SEO, 2)
        formation_pred = round(formation_pred, 2)
        
        
        # Return the prediction as JSON response
        return jsonify({
            'Gestion et coordination du projet': gestion_coordination,
            'Analyste concepteur': analyste_concepteur,
            'Infographie ': prediction_Infographie,
            'Intégration': prediction_Integration,
            'Insertion contenu': prediction_IC,
            'Ingénieur test': prediction_IT,
            'Ingénieur système': prediction_IS,
            'Consultant SEO': prediction_SEO,
            'Formation': formation_pred
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Main function to run the application
if __name__ == '__main__':
    app.run(debug=True)
