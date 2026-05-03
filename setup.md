# AKM MVP — Setup Guide

## Folder Structure

```
akm/
├── SETUP.md
├── requirements.txt
├── .env
├── config.py
├── database.py
├── refiner.py
├── critic.py
├── pipeline.py
└── main.py
```

---

## Step 1 — Install Python

Download Python 3.10 or higher from https://python.org/downloads

Verify it installed correctly by opening your terminal and running:
```
python --version
```
You should see something like `Python 3.11.x`

---

## Step 2 — Create a Virtual Environment

Inside your `akm` folder, run:

```bash
python -m venv venv
```

Then activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You'll know it's active when you see `(venv)` at the start of your terminal line.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This will take a minute — it downloads ChromaDB, the Gemini SDK, OpenAI SDK, and a few utilities.

---

## Step 4 — Add Your API Keys

Open the `.env` file and paste your keys:

```
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
```

Get your Gemini key at: https://aistudio.google.com/app/apikey  
Get your OpenAI key at: https://platform.openai.com/api-keys

---

## Step 5 — Run the System

```bash
python main.py
```

On first run the database will be empty. Type `seed` to load two starter documents, then start asking questions.

---

## What Happens When You Run It

```
You: seed
[SEED] Added 2 documents to the knowledge base.

You: Python also supports async programming with asyncio
[AKM] Retrieved: <node_id>
[AKM] Refining...
[AKM] Mutation Score: 0.88
[AKM] Similarity Score: 0.91
[AKM] Passed gate. Storing as candidate...
[AKM] New candidate stored: <candidate_id>

AKM: Python is a high-level, interpreted programming language...
```

The document won't change yet — it needs 2 confirmations from different sessions before it promotes. Run `main.py` again in a new terminal (different session) and ask the same thing to trigger promotion.

---

## Resetting the Database

Delete the `akm_db/` folder that gets created when you first run the system. It will be recreated fresh on next run.