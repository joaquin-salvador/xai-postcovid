# webapps/CS199/shapash_dashboard/utils/counterfactuals.py
import alibi
from alibi.explainers import CounterfactualProto
import numpy as np
import pandas as pd
from typing import Dict


def initialize_alibi(model, X_train):
    """Initialize Alibi Counterfactual explainer"""
    # Calculate mean and std from training data for feature scaling
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Initialize explainer
    cf = CounterfactualProto(
        predictor=model.predict_proba,
        shape=(1, X_train.shape[1]),
        kappa=1.0,
        beta=0.1,
        feature_range=(
            X_train.min().values,
            X_train.max().values
        ),
        eps=0.01,
        eps_std=0.1,
        max_iterations=500,
        theta=10.0,
    )

    # Fit the explainer on training data
    cf.fit(X_train)
    return cf


def parse_feature_input(feature_values: str, feature_names: list) -> Dict:
    """Parse input string into feature dictionary"""
    try:
        values = [x.strip() for x in feature_values.split(',')]
        feature_dict = {}
        for val in values:
            key, value = val.split('=')
            key = key.strip()
            if key not in feature_names:
                raise ValueError(f"Invalid feature name: {key}")
            feature_dict[key] = float(value) if '.' in value else int(value)
        return feature_dict
    except Exception as e:
        return None


def generate_counterfactual(model, feature_values: str) -> str:
    """Generate counterfactual explanations"""
    if not feature_values:
        return "Please enter feature values in format: feature1=value1, feature2=value2"

    try:
        # Get feature names from model
        feature_names = model.feature_names_

        # Parse input
        feature_dict = parse_feature_input(feature_values, feature_names)
        if not feature_dict:
            return "Invalid input format. Please use: feature1=value1, feature2=value2"

        # Create query instance
        query = pd.DataFrame([feature_dict])
        query = query.reindex(columns=feature_names, fill_value=0)
        X = query.values

        # Generate counterfactual
        cf_explainer = model.cf_explainer  # Set in app.py
        explanation = cf_explainer.explain(X)

        if explanation.cf is not None:
            # Get the counterfactual instance
            cf_instance = explanation.cf['X']

            # Format the changes
            changes = []
            for i, (name, orig, cf) in enumerate(zip(feature_names, X[0], cf_instance[0])):
                if abs(orig - cf) > 0.001:
                    changes.append(f"{name}: {orig:.2f} → {cf:.2f}")

            # Add prediction probabilities
            orig_pred = model.predict_proba(X)[0]
            cf_pred = model.predict_proba(cf_instance)[0]

            result = [
                "Original prediction probabilities:",
                f"Class 0: {orig_pred[0]:.3f}, Class 1: {orig_pred[1]:.3f}",
                "\nCounterfactual prediction probabilities:",
                f"Class 0: {cf_pred[0]:.3f}, Class 1: {cf_pred[1]:.3f}",
                "\nRequired changes:",
                *changes
            ]

            return "\n".join(result)
        else:
            return "No counterfactual found"

    except Exception as e:
        return f"Error generating counterfactual: {str(e)}"