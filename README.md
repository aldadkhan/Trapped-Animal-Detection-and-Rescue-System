# FUR-REAL — Animal Search & Rescue

Upload or take a photo -> the app automatically detects the animal, predicts the
situation (flood, fire, collapse, ...) and risk level, tells you what to do before
the rescue team arrives, shows a treatment guide per species, and sends the
report + photo **directly** to the rescue team by email.

## Run (Windows / Mac / Linux, Python 3.10-3.12)

```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Or just double-click `run.bat` (Windows) / run `./run.sh` (Mac/Linux).

## Direct email setup (one time)
Edit `.streamlit/secrets.toml`:
- Gmail: enable 2-Step Verification, create an App Password at
  https://myaccount.google.com/apppasswords, put it in `SMTP_PASSWORD`.
- Set `SMTP_USER`, `SMTP_SENDER` to your email, and `RESCUE_TEAM_EMAIL`.
Restart the app. The "Send to rescue team now" button sends instantly with the
photo and report attached — no Outlook / mail app popup.

## What was fixed
- Images were not analyzed: `yolo26n.pt` needs `ultralytics>=8.4`; old versions
  failed silently. Requirements are updated and the detector falls back to
  `yolo11n.pt` / `yolov8n.pt` automatically.
- Webcam snapshot used the server's camera (fails in a browser). Now uses the
  browser camera (`st.camera_input`).
- Analysis now runs automatically when a photo is added (no extra button).
- Faster: large photos are downscaled, the model is cached, live video
  detects every 3rd frame.
- New: situation prediction, risk level, first-aid steps, species treatment
  guide, location field, and direct SMTP email.

## Limitations
Predictions are heuristics from image colors/texture + your checklist. They are
not a veterinary diagnosis. Always confirm in person and call professionals.
