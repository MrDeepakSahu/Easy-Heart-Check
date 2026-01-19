# Easy Heart Check

An easy-to-use, AI-powered heart health assessment app built with Streamlit. It asks simple questions, estimates risk, explains reasons, suggests educational medication information, and generates a downloadable report. It can optionally show a temporary Public URL via ngrok.

## Features
- Guided, no-typing checkup with sliders and buttons
- Heart disease risk estimate with clear percentage
- Personalized “Why?” reasoning based on inputs (BP, cholesterol, chest pain, etc.)
- Educational medication info (not a prescription)
- Downloadable text report
- Optional temporary public link via ngrok

## Quick Start

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Run locally

```bash
streamlit run streamlit_app.py --server.port 8503
```

Open http://localhost:8503

### 3) Optional: Enable Public URL (ngrok)

1. Create a free account at https://dashboard.ngrok.com/signup  
2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken  
3. Set it in your terminal (PowerShell example):

```powershell
$env:NGROK_AUTHTOKEN="YOUR_TOKEN_HERE"
```

4. Run the app again. A Public URL will appear on the home screen. Share this URL to allow access from anywhere while the app is running.

## Project Structure

- `streamlit_app.py` — Main application
- `requirements.txt` — Python dependencies
- `.env.example` — Example environment file for secrets (e.g., NGROK_AUTHTOKEN)
- `.gitignore` — Excludes build artifacts and secrets
- `LICENSE` — Project license (MIT)

## Deployment Options

- Streamlit Community Cloud (recommended for simple hosting)
  - Push this repo to GitHub
  - Create a new app in Streamlit Cloud pointing to `streamlit_app.py`

- Render/Railway/Docker-based
  - Create a Dockerfile and deploy to your preferred service

## Disclaimer

- This app is an educational prototype and **NOT** a medical device.
- It does not provide medical diagnosis or prescriptions.
- Always consult a qualified healthcare professional for medical advice.

## License

MIT — see [LICENSE](file:///c:/Users/LENOVO/OneDrive/Desktop/idt/prototype/LICENSE).

