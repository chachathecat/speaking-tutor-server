import { WebSocketServer } from "ws";
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

// ---------- Env helpers
const env = (k, d) => (process.env[k] ?? d);

// ---------- Vertex AI (us-central1 권장)
const vertex = new VertexAI({
  project: env("GCP_PROJECT_ID", ""),
  location: env("GEMINI_LOCATION", "us-central1"),
});
const GEMINI_MODEL = env("GEMINI_MODEL", "gemini-1.5-flash-001");

// ---------- Speech-to-Text
const sttV2 = new speechV2.SpeechClient();
const sttV1 = new speechV1.SpeechClient();

// ---------- Express 기본
const app = express();
app.use(cors());
app.use(express.json());
app.options("*", cors());

// 업로드는 메모리 저장 (buffer 사용)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

// webm/mp4/aac/ogg/mp3 → WAV(PCM_S16LE, mono, 16k)
function toLinear16(inputPath, outPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .outputOptions(["-ac", "1", "-ar", "16000", "-f", "wav", "-acodec", "pcm_s16le"])
      .on("start", (cmd) => console.log("[ffmpeg] start:", cmd))
      .on("stderr", (line) => console.log("[ffmpeg]", line))
      .on("end", resolve)
      .on("error", reject)
      .save(outPath);
  });
}

app.get("/", (_, res) => res.send("OK speaking tutor server"));
app.get("/healthz", (_, res) => res.json({ ok: true }));

// ---- 작은 유틸: 타임아웃 래퍼
const withTimeout = (p, ms) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

app.post("/stt/score", upload.single("audio"), async (req, res) => {
  let inPath, outPath;
  try {
    // 0) 업로드 유효성
    if (!req.file) return res.status(400).json({ ok: false, error: "audio file is required" });
    if (!req.file.buffer?.length) return res.status(400).json({ ok: false, error: "empty audio upload" });

    // 1) 입력 파라미터
    const target = req.body?.target || "";
    // v2에서는 Recognizer가 언어를 가짐. lang은 코칭 텍스트 언어용으로만 활용
    const langForHints = req.body?.lang || "en-US";

    let hints = [];
    try {
      hints = JSON.parse(req.body?.hints || "[]");
      if (!Array.isArray(hints)) hints = [];
    } catch {
      hints = [];
    }
    const wantSemantic = req.body?.semantic === "1";
    const wantCoach = req.body?.coach === "1";

    // 2) 임시 파일
    const mt = (req.file.mimetype || "").toLowerCase();
    const ext =
      mt.includes("mp4") ? "mp4" :
      mt.includes("aac") ? "aac" :
      mt.includes("3gpp") ? "3gp" :
      mt.includes("mpeg") ? "mp3" :
      mt.includes("ogg") ? "ogg" :
      mt.includes("webm") ? "webm" : "dat";

    inPath = path.join(tmpdir(), `${randomUUID()}.${ext}`);
    outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
    await fs.writeFile(inPath, req.file.buffer);

    // 3) WAV 변환
    await toLinear16(inPath, outPath);
    const wavBytes = await fs.readFile(outPath);
    if (!wavBytes.length) {
      return res.status(400).json({ ok: false, error: "wav conversion produced empty audio" });
    }
    const wavB64 = wavBytes.toString("base64");

    // 4) STT 호출: v2 → (옵션) v1 폴백
    const recognizer =
      `projects/${env("GCP_PROJECT_ID", "")}/locations/${env("GCP_LOCATION", "global")}/recognizers/${env("GCP_RECOGNIZER_ID", "")}`;

    let transcript = "";

    try {
      // v2 (권장) — 언어/모델은 Recognizer 설정에 따름
      const v2req = {
        recognizer,
        config: {
          autoDecodingConfig: {}, // 자동 디코딩
          features: { enableAutomaticPunctuation: true },
          ...(hints.length
            ? { adaptation: { phraseSets: [{ phrases: hints.map(v => ({ value: v })), boost: 20.0 }] } }
            : {}),
        },
        content: wavB64,
      };
      const [resp] = await sttV2.recognize(v2req);
      transcript = (resp.results || [])
        .map((r) => r.alternatives?.[0]?.transcript || "")
        .join(" ")
        .trim();
    } catch (e) {
      console.error("[v2 failed]", e?.message || e);

      // v1 폴백은 기본 비활성화(환경변수로 제어)
      const USE_V1_FALLBACK = env("USE_V1_FALLBACK", "0") === "1";
      if (USE_V1_FALLBACK) {
        try {
          const v1req = {
            config: {
              languageCode: langForHints, // v1은 언어가 필요
              enableAutomaticPunctuation: true,
              ...(hints.length ? { speechContexts: [{ phrases: hints }] } : {}), // 빈 배열 금지
            },
            audio: { content: wavB64 },
          };
          const [resp1] = await sttV1.recognize(v1req);
          transcript = (resp1.results || [])
            .map((r) => r.alternatives?.[0]?.transcript || "")
            .join(" ")
            .trim();
        } catch (e1) {
          console.error("[v1 fallback failed]", e1?.message || e1);
          throw e1; // 최종 에러로 처리
        }
      }
    }

    // 5) 스코어링 (타깃 없어도 안전)
    const scoring = scoreAttempt({ transcript, targetText: target });

    // 6) (옵션) 임베딩 유사도 — 6초 제한, 실패 시 스킵
    let semantic = null;
    if (wantSemantic && target && transcript) {
      try {
        const embedder = vertex.getEmbeddingsModel({ model: "text-embedding-004" });
        const [refE, hypE] = await withTimeout(
          Promise.all([
            embedder.embedContent({ content: { parts: [{ text: target }] } }),
            embedder.embedContent({ content: { parts: [{ text: transcript }] } }),
          ]),
          6000
        );
        const a = refE.embeddings[0].values;
        const b = hypE.embeddings[0].values;
        const dot = a.reduce((s, v, i) => s + v * b[i], 0);
        const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
        const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
        const cos = dot / (na * nb);
        semantic = Math.round(Math.max(0, Math.min(1, cos)) * 100);
      } catch (err) {
        console.error("[embedding skipped]", err?.message || err);
      }
    }

    // 7) (옵션) Gemini 코칭 — 6초 제한, 실패 시 스킵
    let aiFeedback = null;
    if (wantCoach) {
      try {
        const gen = vertex.getGenerativeModel({ model: GEMINI_MODEL });
        const prompt = `You are an ESL speaking coach.
Return a compact JSON object with these keys only:
- concise_feedback: array(max 3 bullets)
- grammar_fixes: array of 3 short objects {before, after}
- pronunciation_tips: array(max 2)
- next_prompt: one short line
Language for the user-facing text: ${langForHints}
Reference: """${target}"""
Transcript: """${transcript}"""`;

        const r = await withTimeout(
          gen.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] }),
          6000
        );
        const text = r.response?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
        try {
          aiFeedback = JSON.parse(text);
        } catch {
          aiFeedback = { raw: text };
        }
      } catch (err) {
        console.error("[gemini skipped]", err?.message || err);
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
