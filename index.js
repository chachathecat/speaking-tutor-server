// index.js
import express from "express";
import cors from "cors";
import multer from "multer";

import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";

import { promises as fs } from "fs";
import { tmpdir } from "os";
import path from "path";
import { randomUUID } from "crypto";

import { v2 as speechV2, v1 as speechV1 } from "@google-cloud/speech";
import { scoreAttempt } from "./score.js";

import { VertexAI } from "@google-cloud/vertexai";

// ---------- FFmpeg (Render 등에서 필수)
if (ffmpegPath) ffmpeg.setFfmpegPath(ffmpegPath);

// ---------- Vertex AI
const vertex = new VertexAI({
  project: process.env.GCP_PROJECT_ID,
  location: process.env.GCP_LOCATION,
});

// ---------- Speech-to-Text
const sttV2 = new speechV2.SpeechClient(); // v2 고정
// v1 폴백용은 실패했을 때 new 해도 되지만, 한 번만 만들어 재사용
const sttV1 = new speechV1.SpeechClient();

// ---------- Express 기본
const app = express();
app.use(cors());
app.use(express.json());
app.options("*", cors());

// 업로드는 반드시 메모리 저장 (buffer 사용)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

// webm/mp4/aac/ogg/mp3 → WAV(PCM_S16LE, mono, 16k)
function toLinear16(inputPath, outPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions(["-ac", "1", "-ar", "16000", "-f", "wav", "-acodec", "pcm_s16le"])
      .on("start", cmd => console.log("[ffmpeg] start:", cmd))
      .on("stderr", line => console.log("[ffmpeg]", line))
      .on("end", resolve)
      .on("error", reject)
      .save(outPath);
  });
}

app.get("/", (_, res) => res.send("OK speaking tutor server"));
app.get("/healthz", (_, res) => res.json({ ok: true }));

function cosine(a, b) {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (na * nb);
}

app.post("/stt/score", upload.single("audio"), async (req, res) => {
  let inPath, outPath;
  try {
    // 0) 업로드 유효성
    if (!req.file) return res.status(400).json({ ok: false, error: "audio file is required" });
    if (!req.file.buffer?.length) return res.status(400).json({ ok: false, error: "empty audio upload" });

    // 1) 입력 파라미터
    const target = req.body?.target || "";
    const lang = req.body?.lang || "en-US";
    let hints = [];
    try {
      hints = JSON.parse(req.body?.hints || "[]");
      if (!Array.isArray(hints)) hints = [];
    } catch { hints = []; }
    const wantSemantic = req.body?.semantic === "1";
    const wantCoach    = req.body?.coach === "1";

    // 2) 임시 파일
    const mt = (req.file.mimetype || "").toLowerCase();
    const ext =
      mt.includes("mp4")   ? "mp4"  :
      mt.includes("aac")   ? "aac"  :
      mt.includes("3gpp")  ? "3gp"  :
      mt.includes("mpeg")  ? "mp3"  :
      mt.includes("ogg")   ? "ogg"  :
      mt.includes("webm")  ? "webm" : "dat";

    inPath  = path.join(tmpdir(), `${randomUUID()}.${ext}`);
    outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
    await fs.writeFile(inPath, req.file.buffer);

    // 3) WAV 변환
    await toLinear16(inPath, outPath);
    const wavBytes = await fs.readFile(outPath);
    if (!wavBytes.length) {
      return res.status(400).json({ ok: false, error: "wav conversion produced empty audio" });
    }
    const wavB64 = wavBytes.toString("base64");

    // 4) STT 호출: v2 → 실패 시 v1 폴백
    const recognizer = `projects/${process.env.GCP_PROJECT_ID}/locations/${process.env.GCP_LOCATION}/recognizers/${process.env.GCP_RECOGNIZER_ID}`;
    let transcript = "";

    try {
      // ---- v2 (권장) ----
      const [resp] = await sttV2.recognize({
        recognizer,
        config: {
          languageCode: lang,
          model: "latest_short",
          adaptation: hints.length
            ? { phraseSets: [{ phrases: hints.map(v => ({ value: v })), boost: 20.0 }] }
            : undefined,
        },
        content: wavB64, // v2는 content 필드 (RecognitionAudio 아님)
      });

      transcript = (resp.results || [])
        .map(r => r.alternatives?.[0]?.transcript || "")
        .join(" ")
        .trim();

    } catch (e) {
      console.error("[v2 failed -> fallback v1]", e?.message || e);

      // ---- v1 (폴백) ----
      // v1은 recognizer가 없고, audio:{content} + speechContexts 사용
      const [resp1] = await sttV1.recognize({
        config: {
          languageCode: lang,
          enableAutomaticPunctuation: true,
          speechContexts: hints.length ? [{ phrases: hints, boost: 20.0 }] : [],
        },
        audio: { content: wavB64 }, // v1형식: RecognitionAudio
      });

      transcript = (resp1.results || [])
        .map(r => r.alternatives?.[0]?.transcript || "")
        .join(" ")
        .trim();
    }

    // 5) 스코어링
    const scoring = scoreAttempt({ transcript, targetText: target });

    // 6) (옵션) 임베딩 유사도
    let semantic = null;
    if (wantSemantic && target && transcript) {
      try {
        const embed = vertex.getEmbeddingsModel({ model: "text-embedding-004" });
        const [refE, hypE] = await Promise.all([
          embed.embedContent({ content: { parts: [{ text: target }] } }),
          embed.embedContent({ content: { parts: [{ text: transcript }] } }),
        ]);
        const a = refE.embeddings[0].values;
        const b = hypE.embeddings[0].values;
        semantic = Math.round(Math.max(0, Math.min(1, cosine(a, b))) * 100);
      } catch (err) {
        console.error("[embedding]", err);
      }
    }

    // 7) (옵션) Gemini 코칭
    let aiFeedback = null;
    if (wantCoach) {
      try {
        const gen = vertex.getGenerativeModel({ model: "gemini-2.0-flash" });
        const prompt = `You are an ESL speaking coach.
Return a compact JSON object with these keys only:
- concise_feedback: array(max 3 bullets)
- grammar_fixes: array of 3 short objects {before, after}
- pronunciation_tips: array(max 2)
- next_prompt: one short line
Language for the user-facing text: ${lang}
Reference: """${target}"""
Transcript: """${transcript}"""`;

        const r = await gen.generateContent({
          contents: [{ role: "user", parts: [{ text: prompt }] }],
        });
        const text = r.response?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
        try { aiFeedback = JSON.parse(text); } catch { aiFeedback = { raw: text }; }
      } catch (err) {
        console.error("[gemini]", err);
      }
    }

    // 8) 응답
    res.json({ ok: true, transcript, scoring, semantic, aiFeedback });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: String(e) });
  } finally {
    try { if (inPath) await fs.unlink(inPath); } catch {}
    try { if (outPath) await fs.unlink(outPath); } catch {}
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log("Server on " + PORT));
