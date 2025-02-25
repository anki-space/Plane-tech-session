import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import torch
import random
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import time

# Load dataset
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare Answer Mode dataset
answer_data = {item["question"]: item["answer"] for item in data["answer_mode"]}
questions = list(answer_data.keys())
answers = list(answer_data.values())

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all questions into embeddings
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Function to get the best-matching answer using NLP
def get_best_answer(user_question):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_match_idx].item()

    if best_score > 0.5:  # Threshold for similarity
        return answers[best_match_idx]
    return "Sorry, I don't know the answer to that question."

# Prepare Quiz Mode dataset
quiz_mode_data = data["quiz_mode"]
mode = "answer"
current_question = None
correct_answer = None
score = 0
total_questions = 0

# Function to get timestamp
def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")  # Example: 14:30:15

# Function to switch modes
def switch_mode():
    global mode
    mode = "quiz" if mode == "answer" else "answer"
    mode_label.config(text=f"Mode: {mode.capitalize()} Mode")
    chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: Switched to {mode.capitalize()} Mode.\n", "bot")
    chat_area.yview(tk.END)

# Function to handle user input
def handle_input(event=None):  # Add event argument for Enter key support
    global current_question, correct_answer, score, total_questions
    user_input = user_entry.get().strip()

    if not user_input:
        return

    chat_area.insert(tk.END, f"\n[{get_timestamp()}] [You]: {user_input}\n", "user")
    user_entry.delete(0, tk.END)

    if mode == "answer":
        if user_input.lower() in ["exit", "quit", "bye"]:
            chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: Goodbye! Have a nice day.\n", "bot")
            root.after(1000, root.destroy)  # Close the window after 1 second
        else:
            response = get_best_answer(user_input)
            chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: {response}\n", "bot")

    elif mode == "quiz":
        if current_question:
            if user_input.lower() == correct_answer.lower():
                chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: ✅ Correct!\n", "bot")
                score += 1
            else:
                chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: ❌ Wrong! The correct answer was: {correct_answer}\n", "bot")

            total_questions += 1
            chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: Score: {score}/{total_questions}\n", "bot")
            current_question = None
            correct_answer = None
        else:
            question_data = random.choice(quiz_mode_data)
            current_question = question_data["question"]
            correct_answer = question_data["correct_answer"]
            options = "\n".join(f"- {opt}" for opt in question_data["options"])
            chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: {current_question}\nOptions:\n{options}\n", "bot")

    chat_area.yview(tk.END)


# Function to save chat history
def save_chat():
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS
    filename = f"chat_history_{timestamp}.txt"  # Unique filename with timestamp
    with open(filename, "w", encoding="utf-8") as file:
        file.write(chat_area.get("1.0", tk.END))
    chat_area.insert(tk.END, f"\n[{get_timestamp()}] [Bot]: Chat saved as 'chat_history.txt'!\n", "bot")

# GUI Setup
root = tk.Tk()
root.title("Python Chatbot")
root.geometry("600x600")
root.configure(bg="#2C2F33")

# Chat display area (Styled)
chat_frame = ttk.Frame(root, padding=10)
chat_frame.pack(pady=10, fill="both", expand=True)

chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=70, height=20, bg="#23272A", fg="white", font=("Arial", 12), bd=0, relief=tk.FLAT)
chat_area.pack(padx=5, pady=5, fill="both", expand=True)
chat_area.insert(tk.END, "[Bot]: Hello! Ask me anything about Python or switch to Quiz Mode.\n", "bot")

# User input area
input_frame = ttk.Frame(root, padding=10)
input_frame.pack(fill="x", pady=5)

user_entry = ttk.Entry(input_frame, width=50, font=("Arial", 12))
user_entry.pack(side=tk.LEFT, padx=5, pady=5, fill="x", expand=True)
user_entry.bind("<Return>", handle_input)  # Press Enter to send message

# Button Styling
style = ttk.Style()
style.configure("TButton", font=("Arial", 11), padding=6, relief="raised", borderwidth=3)
style.configure("Send.TButton", background="#4CAF50" )  # Green for Send
style.map("Send.TButton", background=[("active", "#388E3C")])
style.configure("Mode.TButton", background="#FF9800")  # Orange for Mode
style.map("Mode.TButton", background=[("active", "#F57C00")])

# Buttons
send_button = ttk.Button(input_frame, text="Send", command=handle_input, style="Send.TButton")
send_button.pack(side=tk.RIGHT, padx=5)

mode_frame = ttk.Frame(root, padding=5)
mode_frame.pack(fill="x")

mode_label = ttk.Label(mode_frame, text="Mode: Answer Mode", font=("Arial", 12, "bold"), background="#2C2F33", foreground="white")
mode_label.pack(side=tk.LEFT, padx=10)

mode_button = ttk.Button(mode_frame, text="Switch Mode", command=switch_mode, style="Mode.TButton")
mode_button.pack(side=tk.RIGHT, padx=10)


save_button = ttk.Button(mode_frame, text="Save Chat", command=save_chat, style="TButton")
save_button.pack(side=tk.RIGHT, padx=10)

# Chat Text Colors
chat_area.tag_config("bot", foreground="#7289DA")  # Blue for bot
chat_area.tag_config("user", foreground="#43B581")  # Green for user

# Run GUI
root.mainloop()
