# webapps/CS199/shapash_dashboard/components/callbacks.py
from dash.dependencies import Input, Output, State
from ..utils.counterfactuals import generate_counterfactual

def register_callbacks(app, explainer, model):
    @app.callback(
        Output('counterfactual-output', 'children'),
        [Input('generate-button', 'n_clicks')],
        [State('feature-input', 'value')]
    )
    def update_counterfactual(n_clicks, feature_values):
        if n_clicks is None:
            return ""
        return generate_counterfactual(model, feature_values)

    @app.callback(
        Output('shapash-plot', 'figure'),
        [Input('generate-button', 'n_clicks')]
    )
    def update_plot(n_clicks):
        return explainer.plot.contribution_plot(0)