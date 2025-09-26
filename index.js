import express from "express";
import multer from "multer";
import cors from "cors";
import ffmpeg from "fluent-ffmpeg";
import { promises as fs } from "fs";
import { tmpdir } from "os";
import path from "path";
import { randomUUID } from "crypto";
import { SpeechClient } from "@google-cloud/speech";
import { scoreAttempt } from "./score.js";

const app = express();
app.use(cors());

const upload = multer({ limits: { fileSize: 20 * 1024 * 1024 } }); // 20MB
const speech = new SpeechClient(); // uses GOOGLE_APPLICATION_CREDENTIALS

function toLinear16(inputPath, outPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions(["-ac", "1", "-ar", "16000", "-f", "wav", "-acodec", "pcm_s16le"])
      .save(outPath)
      .on("end", resolve)
      .on("error", reject);
  });
}

app.get("/", (_, res) => res.send("OK speaking tutor server"));

app.post("/stt/score", upload.single("audio"), async (req, res) => {
  try {
    const target = req.body?.target || "";
    const inPath = path.join(tmpdir(), `${randomUUID()}.webm`);
    const outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
    await fs.writeFile(inPath, req.file.buffer);

    await toLinear16(inPath, outPath);
    const wavBytes = await fs.readFile(outPath);

    const name = `projects/${process.env.GCP_PROJECT_ID}/locations/${process.env.GCP_LOCATION}/recognizers/${process.env.GCP_RECOGNIZER_ID}`;
    const [resp] = await speech.recognize({
      recognizer: name,
      config: { languageCode: "en-US", model: "short" }, // change language per lesson
      content: wavBytes.toString("base64"),
    });

    const transcript = (resp.results || [])
      .map(r => (r.alternatives?.[0]?.transcript || ""))
      .join(" ")
      .trim();

    const scoring = scoreAttempt({ transcript, targetText: target });

    await fs.unlink(inPath).catch(()=>{});
    await fs.unlink(outPath).catch(()=>{});

    res.json({ ok: true, transcript, scoring });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: String(e) });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log("Server on " + PORT));
