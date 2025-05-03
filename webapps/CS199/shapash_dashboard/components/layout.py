# webapps/CS199/shapash_dashboard/components/layout.py
from dash import dcc, html

def create_layout():
    return html.Div([
        html.Div([
            html.H1("Model Explanations Dashboard"),
            html.Div([
                html.Div([
                    dcc.Graph(id='shapash-plot')
                ], style={'width': '70%', 'display': 'inline-block'}),
                html.Div([
                    html.H3("Counterfactual Generation"),
                    html.Div([
                        html.Label("Feature Values:"),
                        dcc.Input(id='feature-input', type='text', placeholder='Enter values'),
                        html.Button('Generate', id='generate-button'),
                        html.Div(id='counterfactual-output')
                    ])
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
            ])
        ]),
        # Add to your layout file
        html.Div([
            html.Label('Target Value'),
            dcc.Input(
                id='target-value-input',
                type='number',
                placeholder='Enter desired prediction'
            )
        ])
    ])