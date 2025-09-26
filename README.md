# Speaking Tutor Server (Render-ready)

Minimal Node/Express server that receives webm/opus audio, converts to WAV (LINEAR16 16k mono), calls Google Cloud Speech-to-Text v2, and returns a simple scoring result.

## Deploy on Render (Free)

1. Push this folder to a new GitHub repo.
2. On Render: New → Web Service → Connect the repo.
3. Build Command: `npm i`
4. Start Command: `node index.js`
5. Environment Variables:
   - `GCP_PROJECT_ID` = your GCP project id
   - `GCP_LOCATION` = e.g., `asia-northeast3`
   - `GCP_RECOGNIZER_ID` = your recognizer id (e.g., `vlx-en-rec-001`)
   - `GOOGLE_APPLICATION_CREDENTIALS` = `/etc/secrets/gcp.json`
6. Secret Files:
   - Path: `/etc/secrets/gcp.json`
   - Content: paste your Service Account JSON

After deploy, your endpoint will be:
```
POST https://<your-app>.onrender.com/stt/score
FormData:
  audio: <webm blob>
  target: <reference sentence>
```
