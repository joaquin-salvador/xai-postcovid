# webapps/CS199/shapash_dashboard/components/callbacks.py
from dash.dependencies import Input, Output, State
from ..utils.counterfactuals import generate_counterfactual
import pandas as pd

def register_callbacks(app, explainer):
    @app.callback(
        Output('shapash-plot', 'figure'),
        [Input('generate-button', 'n_clicks')],
        [State('feature-input', 'value')]
    )
    def update_shapash_plot(n_clicks, feature_values):
        if n_clicks is None:
            return {}
        # Shapash plot logic here
        return {}

    @app.callback(
        Output('counterfactual-output', 'children'),
        [Input('generate-button', 'n_clicks')],
        [State('feature-input', 'value'),
         State('target-value-input', 'value')]
    )
    def update_counterfactual(n_clicks, feature_values, target_value):
        if n_clicks is None:
            return "Click the button to generate counterfactual"

        if feature_values is None or target_value is None:
            return "Please enter all values"

        try:
            # Convert feature values to DataFrame
            instance = pd.DataFrame([feature_values])

            # Generate counterfactual
            result = generate_counterfactual(explainer.cf_explainer, instance, target_value)

            if result is None or not result['success']:
                return "Could not find valid counterfactual"

            # Format the result
            cf_values = pd.DataFrame(result['counterfactual'])
            original = pd.DataFrame(result['original'])

            return f"Original prediction: {explainer.model.predict(original)[0]:.2f}\n" \
                   f"Counterfactual prediction: {explainer.model.predict(cf_values)[0]:.2f}"

        except Exception as e:
            return f"Error: {str(e)}"