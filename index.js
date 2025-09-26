// index.js
import express from "express";
import multer from "multer";
import cors from "cors";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";          // ★ ffmpeg 바이너리 포함
import { promises as fs } from "fs";
import { tmpdir } from "os";
import path from "path";
import { randomUUID } from "crypto";
import { SpeechClient } from "@google-cloud/speech";
import { scoreAttempt } from "./score.js";

// ── Vertex AI (임베딩 + Gemini 코칭)
import { VertexAI } from "@google-cloud/vertexai";
const vertex = new VertexAI({
  project: process.env.GCP_PROJECT_ID,
  location: process.env.GCP_LOCATION, // e.g., "asia-northeast3"
});

// ── ffmpeg 경로 지정 (Render 등에서 필수)
if (ffmpegPath) ffmpeg.setFfmpegPath(ffmpegPath);

const app = express();

// ── 기본 미들웨어
app.use(cors());
app.use(express.json());
app.options("*", cors());

// ── 업로드: 반드시 메모리 저장 사용! (buffer로 받기 위함)
const upload = multer({
  storage: multer.memoryStorage(),              // ★ 중요
  limits: { fileSize: 20 * 1024 * 1024 },
});

const speech = new SpeechClient(); // GOOGLE_APPLICATION_CREDENTIALS 사용

// ── webm/mp4/aac/ogg/mp3 → WAV(PCM_S16LE, mono, 16k)
function toLinear16(inputPath, outPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions(["-ac", "1", "-ar", "16000", "-f", "wav", "-acodec", "pcm_s16le"])
      .on("start", cmd => console.log("[ffmpeg] start:", cmd))
      .on("stderr", line => console.log("[ffmpeg]", line))
      .on("end", resolve)
      .on("error", err => {
        console.error("[ffmpeg error]", err);
        reject(err);
      })
      .save(outPath);
  });
}

// ── 헬스 체크
app.get("/", (_, res) => res.send("OK speaking tutor server"));
app.get("/healthz", (_, res) => res.json({ ok: true }));

// ── 유틸: 코사인 유사도
function cosine(a, b) {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0);
  const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return dot / (na * nb);
}

// ── STT + 스코어 + (옵션) 임베딩/코칭
app.post("/stt/score", upload.single("audio"), async (req, res) => {
  let inPath, outPath;
  try {
    // 0) 업로드 유효성
    if (!req.file) {
      return res.status(400).json({ ok: false, error: "audio file is required" });
    }
    if (!req.file.buffer || !req.file.buffer.length) {
      return res.status(400).json({ ok: false, error: "empty audio upload" });
    }

    // 1) 입력 파라미터
    const target = req.body?.target || "";
    const lang = req.body?.lang || "en-US"; // 'en-US' | 'ko-KR' ...
    let hints = [];
    try {
      hints = JSON.parse(req.body?.hints || "[]");
      if (!Array.isArray(hints)) hints = [];
    } catch { hints = []; }
    const wantSemantic = req.body?.semantic === "1"; // 임베딩 유사도
    const wantCoach    = req.body?.coach === "1";    // Gemini 코칭

    // 2) 업로드 포맷 식별 → 임시 파일 저장
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
    console.log("[upload]", req.file.mimetype, req.file.size, "->", inPath);

    // 3) WAV 변환(방어)
    try {
      await toLinear16(inPath, outPath);
    } catch (err) {
      return res.status(400).json({
        ok: false,
        error: "audio conversion failed: " + (err?.message || err),
      });
    }

    const wavBytes = await fs.readFile(outPath);
    if (!wavBytes.length) {
      return res.status(400).json({ ok: false, error: "wav conversion produced empty audio" });
    }

    console.log("[stt] lang=%s hints=%d bytes=%d", lang, hints.length, wavBytes.length);

    // 4) Google STT v2 (latest_short + phrase hints)
    const recognizer =
      `projects/${process.env.GCP_PROJECT_ID}/locations/${process.env.GCP_LOCATION}/recognizers/${process.env.GCP_RECOGNIZER_ID}`;

    const [resp] = await speech.recognize({
      recognizer,
      config: {
        languageCode: lang,
        model: "latest_short",
        adaptation: hints.length
          ? { phraseSets: [{ phrases: hints.map(v => ({ value: v })), boost: 20.0 }] }
          : undefined,
        // 필요 시: autoDecodingConfig: {},
      },
      content: wavBytes.toString("base64"),
    });

    const transcript = (resp.results || [])
      .map(r => r.alternatives?.[0]?.transcript || "")
      .join(" ")
      .trim();

    // 5) 규칙 기반 스코어
    const scoring = scoreAttempt({ transcript, targetText: target });

    // 6) (옵션) 의미 유사도: Vertex Embedding
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

    // 7) (옵션) Gemini 코칭 피드백
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
          aiFeedback = { raw: text }; // 모델이 JSON이 아닌 경우 대비
        }
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
    if (inPath) await fs.unlink(inPath).catch(() => {});
    if (outPath) await fs.unlink(outPath).catch(() => {});
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log("Server on " + PORT));
