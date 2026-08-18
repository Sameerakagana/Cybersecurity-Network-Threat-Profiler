CYBERSECURITY NETWORK THREAT & INTRUSION PROFILER
==================================================

This project uses:
- FastAPI backend
- HTML/CSS/JavaScript frontend
- Random Forest classification
- Isolation Forest anomaly detection
- NSL-KDD dataset

FOLDER STRUCTURE
----------------
data/
    PUT_YOUR_NSL_KDD_CSV_HERE.txt

models/
    Model files are created automatically after training.

templates/
    index.html

static/
    style.css
    script.js

HOW TO RUN IN VS CODE
---------------------

1. Open this project folder in VS Code.

2. Open Terminal > New Terminal.

3. Create a virtual environment:
   python -m venv venv

4. Activate it on Windows:
   venv\Scripts\activate

5. Install dependencies:
   pip install -r requirements.txt

6. Put your NSL-KDD CSV inside the data folder.
   IMPORTANT: Rename it to:
   NSL_KDD.csv

   Final location:
   data/NSL_KDD.csv

7. Train the ML models:
   python train_model.py

8. Start the website:
   uvicorn app:app --reload

9. Open in Chrome:
   http://127.0.0.1:8000

IMPORTANT
---------
The dataset itself is intentionally not included because the project
cannot redistribute a third-party dataset file. The data folder is
already included so you only need to place your NSL_KDD.csv inside it.

The training script accepts a target column named:
- label
- class
- attack

It automatically detects categorical columns and saves the feature
configuration used by the model.

For a real deployment, network traffic should be collected through an
authorized monitoring system and fed into the same feature pipeline.
