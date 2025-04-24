# webapps/CS199/shapash_dashboard/app.py
import dash
from dash import html
import pandas as pd
import joblib
from .components.layout import create_layout
from .components.callbacks import register_callbacks
from .explainer.model import load_model, get_feature_names
from .explainer.explainer import create_explainer
from .utils.counterfactuals import initialize_alibi


def create_app():
    app = dash.Dash(__name__)

    # Load model and scaler
    model = load_model()
    scaler = joblib.load('../scaler.joblib')

    # Load training data
    X_train = pd.read_csv('../training_data.csv')

    # Scale the training data
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Store feature names in model
    model.feature_names_ = get_feature_names()

    # Create explainers
    explainer = create_explainer(model, X_train_scaled)
    model.cf_explainer = initialize_alibi(model, X_train_scaled)

    # Set up the layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app, explainer, model)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)