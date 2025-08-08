import tkinter as tk
from tkinter import messagebox
import subprocess
import pandas as pd
import os

# Function to start attendance (runs face_recognition.py)
def start_attendance():
    try:
        subprocess.run(["python", "face_recognition.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Unable to start attendance: {e}")

# Function to display attendance list
def show_attendance():
    try:
        if not os.path.exists("recognized_faces.txt"):
            messagebox.showinfo("Info", "No attendance data available.")
            return

        with open("recognized_faces.txt", "r") as f:
            data = f.read()

        top = tk.Toplevel(root)
        top.title("Attendance List")
        top.geometry("600x400")

        text = tk.Text(top)
        text.pack(expand=True, fill='both')
        text.insert(tk.END, data)

    except Exception as e:
        messagebox.showerror("Error", f"Unable to open recognized_faces.txt: {e}")

# Function to exit the application
def quit_app():
    root.destroy()

# Create main window
root = tk.Tk()
root.title("Student Attendance System using Face Recognition")
root.geometry("400x300")
root.configure(bg="white")

# Title label
title = tk.Label(root, text="Student Attendance System", font=("Arial", 20, "bold"), bg="white", fg="blue")
title.pack(pady=20)

# Start attendance button
btn_start = tk.Button(root, text="Start Attendance", font=("Arial", 14), width=25, bg="#4CAF50", fg="white", command=start_attendance)
btn_start.pack(pady=10)

# View attendance list button
btn_view = tk.Button(root, text="View Attendance List", font=("Arial", 14), width=25, bg="#2196F3", fg="white", command=show_attendance)
btn_view.pack(pady=10)

# Exit button
btn_quit = tk.Button(root, text="Exit", font=("Arial", 14), width=25, bg="#f44336", fg="white", command=quit_app)
btn_quit.pack(pady=10)

# Start GUI
root.mainloop()