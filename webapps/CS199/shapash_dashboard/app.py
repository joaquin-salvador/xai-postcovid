# webapps/CS199/shapash_dashboard/app.py
import os
import dash
import pandas as pd
import joblib
from .components.layout import create_layout
from .components.callbacks import register_callbacks
from .explainer.model import load_model, get_feature_names
from .explainer.explainer import create_explainer
from .utils.counterfactuals import initialize_alibi

def create_app():
    app = dash.Dash(__name__)
    app.layout = create_layout()

    base_dir = os.path.dirname(os.path.dirname(__file__))

    # Load model
    model = load_model()

    # Load scaler
    scaler_path = os.path.join(base_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)

    # Load training data
    data_path = os.path.join(base_dir, 'training_data.csv')
    X_train = pd.read_csv(data_path)

    # Add debugging code here
    print("Required features:", len(get_feature_names()))
    print("Available features:", X_train.shape[1])
    print("Missing features:", set(get_feature_names()) - set(X_train.columns))

    # Get required features
    required_features = get_feature_names()

    # Select only the required features
    X_train = X_train[required_features]

    # Scale the training data
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Store feature names in model
    model.feature_names_ = required_features

    # Create explainers
    explainer = create_explainer(model, X_train_scaled)
    model.cf_explainer = initialize_alibi(model, X_train_scaled)

    # Set up layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app, model)

    return app