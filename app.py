
## **10. Web Interface with Gradio (10 Marks)**

# Create a user-friendly Gradio web interface that takes user inputs and displays the prediction from your trained model.


import pandas as pd
import gradio as gr
import pickle


# 1. Loaded saved model
with open("water_predict_model.pkl", "rb") as f:
    best_model = pickle.load(f)

def predict_water(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    quality_index = ph * hardness / solids
    input_data = [[float(ph), float(hardness), float(solids), float(chloramines),
                   float(sulfate), float(conductivity), float(organic_carbon),
                   float(trihalomethanes), float(turbidity), float(quality_index)]]
    prediction = best_model.predict(input_data)[0]
    return "Drinkable ✅" if prediction == 1 else "Not Drinkable ❌"


water_predict_app = gr.Interface(
    fn=predict_water,
    inputs=[
        gr.Number(label="pH"),
        gr.Number(label="Hardness"),
        gr.Number(label="Solids"),
        gr.Number(label="Chloramines"),
        gr.Number(label="Sulfate"),
        gr.Number(label="Conductivity"),
        gr.Number(label="Organic Carbon"),
        gr.Number(label="Trihalomethanes"),
        gr.Number(label="Turbidity")
    ],
    outputs="text",
    title="💧 Water Potability Prediction",
    description="Enter water quality parameters to check if it's drinkable."
)

water_predict_app.launch(share = True)


"""## **11. Deployment to Hugging Face (10 Marks)**

Hugging Face Spaces public URL: 
"""