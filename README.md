# Reconciliation Prototype — CM3070 Final Project

This repository contains the full source code and documentation for my CM3070 Computer Science Final Project. 
The project explores the concept of *reconciliation* through a prototype implementation, demonstrating my ability 
to design, build, and evaluate a complete software artefact.

---

# Setup
1. Create a Python 3 venv and activate it:
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Set your OpenAI API key:
   export OPENAI_API_KEY="sk-..."

4. Run the app:
   python app.py

5. Test endpoints (in a second terminal):
   curl http://127.0.0.1:5000/health
   curl "http://127.0.0.1:5000/reconcile?query=London%20England&limit=5"
