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

// 🔸 Vertex AI (임베딩 + Gemini 코칭)
import { VertexAI } from "@google-cloud/vertexai";
const vertex = new VertexAI({
  project: process.env.GCP_PROJECT_ID,
  location: process.env.GCP_LOCATION, // 예: asia-northeast3
});

const app = express();
app.use(cors());

const upload = multer({ limits: { fileSize: 20 * 1024 * 1024 } }); // 20MB
const speech = new SpeechClient(); // GOOGLE_APPLICATION_CREDENTIALS 사용

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

function cosine(a, b) {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (na * nb);
}

app.post("/stt/score", upload.single("audio"), async (req, res) => {
  let inPath, outPath;
  try {
    if (!req.file) {
      return res.status(400).json({ ok: false, error: "audio file is required" });
    }
    if (!req.file.buffer?.length) {
      return res.status(400).json({ ok: false, error: "empty audio upload" });
    }

    // ✅ 입력 파라미터
    const target = req.body?.target || "";
    const lang = req.body?.lang || "en-US";                // 예: 'en-US' / 'ko-KR'
    const hints = JSON.parse(req.body?.hints || "[]");     // 예: ["ship","sheep"]
    const wantSemantic = req.body?.semantic === "1";       // 임베딩 점수
    const wantCoach = req.body?.coach === "1";             // Gemini 코칭

    // 1) 업로드 포맷 식별 → wav(LINEAR16 16k mono) 변환
    const ext =
      req.file.mimetype?.includes("mp4")  ? "mp4"  :
      req.file.mimetype?.includes("mpeg") ? "mp3"  :
      req.file.mimetype?.includes("ogg")  ? "ogg"  :
      req.file.mimetype?.includes("webm") ? "webm" : "dat";
    inPath = path.join(tmpdir(), `${randomUUID()}.${ext}`);
    outPath = path.join(tmpdir(), `${randomUUID()}.wav`);

    await fs.writeFile(inPath, req.file.buffer);
    console.log("[upload]", req.file.mimetype, req.file.size, "->", inPath);

    try {
      await toLinear16(inPath, outPath);
    } catch (convErr) {
      console.error("[ffmpeg]", convErr);
      return res.status(400).json({ ok: false, error: "audio conversion failed" });
    }

    const wavBytes = await fs.readFile(outPath);
    if (!wavBytes.length) {
      return res.status(400).json({ ok: false, error: "wav conversion produced empty audio" });
    }

    // 2) Google STT v2 (latest_short + phrase hints)
    const name = `projects/${process.env.GCP_PROJECT_ID}/locations/${process.env.GCP_LOCATION}/recognizers/${process.env.GCP_RECOGNIZER_ID}`;
    const [resp] = await speech.recognize({
      recognizer: name,
      config: {
        languageCode: lang,
        model: "latest_short",
        adaptation: hints.length
          ? { phraseSets: [{ phrases: hints.map(v => ({ value: v })), boost: 20.0 }] }
          : undefined,
      },
      content: wavBytes.toString("base64"),
    });

    const transcript =
      (resp.results || [])
        .map(r => r.alternatives?.[0]?.transcript || "")
        .join(" ")
        .trim();

    // 3) 규칙 기반 점수 (v0)
    const scoring = scoreAttempt({ transcript, targetText: target });

    // 4) (선택) 의미 유사도: Vertex 임베딩
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
        semantic = Math.round(Math.max(0, Math.min(1, cosine(a, b))) * 100); // 0~100
      } catch (err) {
        console.error("[embedding]", err);
      }
    }

    // 5) (선택) Gemini 코칭 피드백
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
        try {
          aiFeedback = JSON.parse(text);
        } catch {
          aiFeedback = { raw: text }; // 모델이 JSON이 아닌 텍스트를 줄 경우
        }
      } catch (err) {
        console.error("[gemini]", err);
      }
    }

    res.json({ ok: true, transcript, scoring, semantic, aiFeedback });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: String(e) });
  } finally {
    // 임시 파일 정리
    if (inPath) await fs.unlink(inPath).catch(() => {});
    if (outPath) await fs.unlink(outPath).catch(() => {});
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log("Server on " + PORT));
