import tkinter as tk
from tkinter import messagebox, Canvas
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
import winsound  # For alarm sound on Windows

# Load models and scaler
lstm_model = load_model('lstm_model.h5')
lgb_model = lgb.Booster(model_file='lgb_model.txt')
scaler = joblib.load('scaler.pkl')

# Function to predict failure
def predict_failure(input_values):
    input_scaled = scaler.transform([input_values])
    input_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    
    lstm_pred = (lstm_model.predict(input_lstm) > 0.5).astype(int)[0][0]
    lgb_pred = (lgb_model.predict(input_scaled) > 0.5).astype(int)[0]
    
    combined_pred = 1 if (lstm_pred + lgb_pred) > 1 else 0
    
    return lstm_pred, lgb_pred, combined_pred

# Function to plot results with a curve and labels
def plot_results(prediction, canvas_frame):
    colors = {0: 'green', 1: 'red', 2: 'yellow'}
    labels = {0: 'Optimal', 1: 'Failure', 2: 'Moderate'}
    
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) if prediction == 0 else np.cos(x) if prediction == 2 else np.tan(x)
    ax.plot(x, y, color=colors[prediction], label=labels[prediction])
    ax.set_ylim(-2, 2)
    ax.legend()
    ax.set_title('Machine Failure Prediction Curve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Performance Index')
    
    for widget in canvas_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.get_tk_widget().pack()
    canvas.draw()

# Function to trigger alarm sound
def trigger_alarm():
    winsound.Beep(2000, 1000)  # Frequency 2000 Hz, Duration 500ms

# Function to handle prediction button
def on_predict():
    try:
        input_values = [float(entries[i].get()) for i in range(len(labels))]
        lstm_pred, lgb_pred, combined_pred = predict_failure(input_values)
        
        failure_count = lstm_pred + lgb_pred
        prediction = 1 if failure_count > 1 else (2 if failure_count == 1 else 0)
        
        result_text = f"LSTM Prediction: {'Failure' if lstm_pred == 1 else 'Optimal'}\n"
        result_text += f"LightGBM Prediction: {'Failure' if lgb_pred == 1 else 'Optimal'}\n"
        result_text += f"Combined Prediction: {'Failure' if combined_pred == 1 else 'Optimal'}"
        
        if prediction == 1:
            result_text += "\nSuggested Measures: Check machine components, reduce load, perform maintenance, and inspect temperature and torque levels."
            trigger_alarm()
        
        messagebox.showinfo("Prediction Result", result_text)
        plot_results(prediction, canvas_frame)
        
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# UI Setup
root = tk.Tk()
root.title("Machine Failure Prediction")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

labels = ['Type', 'Air_temp', 'Process_temp', 'Rotational_speed', 'Torque', 'Tool_wear', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
entries = []

for i, label in enumerate(labels):
    tk.Label(frame, text=label).grid(row=i, column=0, pady=5, sticky='w')
    entry = tk.Entry(frame)
    entry.grid(row=i, column=1, pady=5)
    entries.append(entry)

predict_button = tk.Button(frame, text="Predict", command=on_predict)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

canvas_frame = tk.Frame(root)
canvas_frame.pack()

root.mainloop()
