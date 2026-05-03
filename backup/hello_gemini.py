# hello_gemini.py
# A simple test to make sure Vertex AI is working

import vertexai
from vertexai.generative_models import GenerativeModel

# ──────────────────────────────────────────────────────────
# Replace these with your actual values
# ──────────────────────────────────────────────────────────
PROJECT_ID = "pdf-rag-vertex"   # ← change this!
LOCATION = "us-central1"         # The region where Vertex AI runs

# ──────────────────────────────────────────────────────────
# Initialize Vertex AI
# ──────────────────────────────────────────────────────────
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ──────────────────────────────────────────────────────────
# Load the Gemini model
# ──────────────────────────────────────────────────────────
model = GenerativeModel("gemini-2.5-flash")

# ──────────────────────────────────────────────────────────
# Send a test question
# ──────────────────────────────────────────────────────────
response = model.generate_content("Say hello and tell me one fun fact about Google Cloud!")

print("\n🤖 Gemini says:")
print(response.text)