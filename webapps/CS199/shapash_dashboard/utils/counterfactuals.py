# webapps/CS199/shapash_dashboard/utils/counterfactuals.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def initialize_alibi(model, X_train):
    """Initialize a simple counterfactual generator"""
    return {
        'model': model,
        'scaler': MinMaxScaler().fit(X_train),
        'feature_ranges': {
            'min': X_train.min(),
            'max': X_train.max()
        }
    }


def generate_counterfactual(explainer, instance, target_value):
    """Generate counterfactual using perturbation-based approach"""
    try:
        # Convert target value to float
        target = float(target_value)

        # Get model and ranges
        model = explainer['model']
        feature_ranges = explainer['feature_ranges']

        # Convert instance to numpy if needed
        if isinstance(instance, pd.DataFrame):
            original = instance.values
        else:
            original = instance.copy()

        # Reshape if needed
        if len(original.shape) == 1:
            original = original.reshape(1, -1)

        # Initialize counterfactual as copy of original
        counterfactual = original.copy()

        # Perturbation parameters
        max_iter = 100
        step_size = 0.1
        tolerance = 0.1

        # Iteratively perturb features
        for _ in range(max_iter):
            pred = model.predict(counterfactual)[0]

            if abs(pred - target) < tolerance:
                return {
                    'counterfactual': counterfactual,
                    'original': original,
                    'success': True
                }

            # Calculate gradient direction
            direction = 1 if pred < target else -1

            # Perturb features within bounds
            for j in range(counterfactual.shape[1]):
                delta = direction * step_size * (feature_ranges['max'].iloc[j] - feature_ranges['min'].iloc[j])
                counterfactual[0, j] = np.clip(
                    counterfactual[0, j] + delta,
                    feature_ranges['min'].iloc[j],
                    feature_ranges['max'].iloc[j]
                )

        return {
            'counterfactual': counterfactual,
            'original': original,
            'success': False
        }

    except Exception as e:
        print(f"Error generating counterfactual: {str(e)}")
        return None