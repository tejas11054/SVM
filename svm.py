import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    global X, y, feature_names, target_names
    filepath = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if filepath:
        entry_filepath.delete(0, tk.END)
        entry_filepath.insert(0, filepath)
        df = pd.read_csv(filepath)
        X = df.drop(columns=[df.columns[-1]]).values
        y = df[df.columns[-1]].values
        feature_names = df.columns[:-1].tolist()
        target_names = df[df.columns[-1]].unique().tolist()

def perform_svm():
    global X, y, feature_names, target_names
    filepath = entry_filepath.get()
    if not filepath:
        messagebox.showerror("Error", "Please select a file first.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = svm.SVC(kernel='linear')
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("SVM Results", f"Accuracy: {accuracy:.2f}")

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create Tkinter window
root = tk.Tk()
root.title("SVM Classifier")
root.geometry("400x200")  # Set window size
root.configure(bg='lightblue')  # Set background color

# Create input field and button
frame = tk.Frame(root, bg='lightblue')  # Set frame background color
frame.pack(padx=20, pady=20)

label_filepath = tk.Label(frame, text="File Path:", bg='lightblue')  # Set label background color
label_filepath.grid(row=0, column=0, sticky="e")

entry_filepath = tk.Entry(frame, width=30)
entry_filepath.grid(row=0, column=1, padx=5, pady=5)

button_browse = tk.Button(frame, text="Browse", command=load_data, bg='lightgray', fg='black')  # Set button background and foreground color
button_browse.grid(row=0, column=2, padx=5, pady=5)

button_calculate = tk.Button(frame, text="Calculate SVM", command=perform_svm, bg='lightgreen', fg='black')  # Set button background and foreground color
button_calculate.grid(row=1, columnspan=3, padx=5, pady=5)

# Define global variables for data
X = None
y = None
feature_names = None
target_names = None

root.mainloop()
