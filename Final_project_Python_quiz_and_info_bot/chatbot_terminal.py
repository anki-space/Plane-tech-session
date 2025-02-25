import json
import random
import torch
from sentence_transformers import SentenceTransformer, util

# Load dataset
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Prepare Answer Mode dataset
answer_data = {item["question"]: item["answer"] for item in data["answer_mode"]}
questions = list(answer_data.keys())
answers = list(answer_data.values())

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all questions into embeddings
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Function to get the best-matching answer using NLP
def get_best_answer(user_question):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_match_idx].item()

    if best_score > 0.5:  # Similarity threshold
        return answers[best_match_idx]
    return "Sorry, I don't know the answer to that question."




# Prepare Quiz Mode dataset
quiz_mode_data = data["quiz_mode"]


# Chatbot loop
def chatbot():
    print("Hello! I'm your Python chatbot. Type 'switch' to change modes or 'exit' to quit.")
    mode = "answer"
    current_question = None
    correct_answer = None

    while True:
        user_input = input("\nYou: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        elif user_input=="mode":
            print(f"Current mode:{mode.capitalize()}")

        elif user_input == "switch":
            mode = "quiz" if mode == "answer" else "answer"
            print(f"Chatbot: Switched to {mode.capitalize()} Mode.")


        elif mode == "answer":
            response = get_best_answer(user_input)
            print(f"Chatbot: {response}")

        elif mode == "quiz":
            if current_question:
                if user_input == correct_answer.lower():
                    print("Chatbot: ✅ Correct!")
                else:
                    print(f"Chatbot: ❌ Wrong! The correct answer was: {correct_answer}")
                current_question = None
                correct_answer = None
            else:
                question_data = random.choice(quiz_mode_data)
                current_question = question_data["question"]
                correct_answer = question_data["correct_answer"]
                options = "\n".join(f"- {opt}" for opt in question_data["options"])
                print(f"Chatbot: {current_question}\nOptions:\n{options}")



# Run chatbot
if __name__ == "__main__":
    chatbot()
