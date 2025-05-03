# webapps/CS199/shapash_dashboard/explainer/model.py
import joblib
import os

def load_model():
    """Load the trained model"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lightgbm_optimized_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

def get_feature_names():
    """Return the exact feature names used during model training"""
    return [
        'age', 'hincome', 'pincome', 'number_fam', 'exercise', 'healthydiet',
        'healthysleep', 'favoriteactivity', 'interaction_offline', 'interaction_online',
        'pb_continuous', 'pb_altruistic', 'pb_avoidant', 'pb_understanding',
        'optimism', 'deteriorationeconomy', 'deteriorationinteract', 'frustration',
        'covidanxiety', 'covidsleepless', 'difficultyliving', 'difficultywork',
        'k6_total', 'phq9_total', 'lsns6_total', 'sss8_total', 'gad7_total',
        'shs_total', 'sex_Male', 'married_Unmarried', 'child_With children',
        'job_group_Homemaker', 'job_group_Others', 'job_group_Student',
        'job_group_Unemployed', 'med_self_Yes', 'medself_covid_Yes', 'med_fam_Yes',
        'medfam_covid_Yes', 'current_physical_Yes', 'past_physical_Yes',
        'current_mental_Yes', 'past_mental_Yes', 'current_covid19_Yes',
        'past_covid19_Yes'
    ]