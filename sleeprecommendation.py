# modules/sleep_recommendation.py
import numpy as np

def get_sleep_recommendations(user_data):
    """
    Generate personalized recommendations based on input features.
    user_data: dict with keys ['Age', 'Stress_Level', 'Physical_Activity', 'Sleep_Duration', 'BMI']
    """
    recommendations = []

    # Sleep duration logic
    if user_data['Sleep_Duration'] < 6:
        recommendations.append("Try to increase sleep duration to at least 7 hours per night.")
    elif user_data['Sleep_Duration'] > 9:
        recommendations.append("Consider reducing oversleeping; consistent 7–9 hours is ideal.")

    # Stress
    if user_data['Stress_Level'] > 6:
        recommendations.append("Practice stress management (mindfulness, breathing, or physical activity).")

    # Physical activity
    if user_data['Physical_Activity'] < 3:
        recommendations.append("Increase physical activity to at least 30 minutes a day, 5 days a week.")

    # BMI
    if user_data['BMI'] > 25:
        recommendations.append("Maintain a balanced diet and consider moderate exercise to manage BMI.")

    if not recommendations:
        recommendations.append("Your lifestyle habits look balanced — keep it up!")

    return recommendations
