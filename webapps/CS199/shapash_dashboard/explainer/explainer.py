# webapps/CS199/shapash_dashboard/explainer/explainer.py
from shapash.explainer.smart_explainer import SmartExplainer

def create_explainer(model, X_train):
    explainer = SmartExplainer(model=model)
    explainer.compile(x=X_train)
    return explainer