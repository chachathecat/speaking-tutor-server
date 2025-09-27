// server.js
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
import textToSpeech from "@google-cloud/text-to-speech";
import { scoreAttempt } from "./score.js";

import { VertexAI } from "@google-cloud/vertexai";

// ---------- FFmpeg (Render 등에서 필수)
if (ffmpegPath) ffmpeg.setFfmpegPath(ffmpegPath);

// ---------- Env helpers (과거/현재 키 병행 지원)
const envAny = (keys, d = "") => {
  for (const k of keys) if (process.env[k] != null) return process.env[k];
  return d;
};

// 통일된 키 (이름은 GOOGLE_* 기본, 예전 GCP_*도 허용)
const PROJECT_ID = envAny(["GOOGLE_PROJECT_ID", "GCP_PROJECT_ID"], "");
const LOCATION   = envAny(["GOOGLE_LOCATION", "GEMINI_LOCATION", "GCP_LOCATION"], "us-central1");
const RECOGNIZER_ID = envAny(["GOOGLE_RECOGNIZER_ID", "GCP_RECOGNIZER_ID"], "");
const GEMINI_MODEL  = envAny(["GEMINI_MODEL"], "publishers/google/models/gemini-1.5-flash");

// ---------- Vertex AI
const vertex = new VertexAI({ project: PROJECT_ID, location: LOCATION });

// ---------- Speech / TTS
const sttV2 = new speechV2.SpeechClient();
const sttV1 = new speechV1.SpeechClient();
const tts = new textToSpeech.TextToSpeechClient();

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

// ---- TTS
function pickVoice(lang = "en-US") {
  // 필요하면 더 추가
  if (lang.startsWith("ko")) return { languageCode: "ko-KR", name: "ko-KR-Neural2-A" };
  if (lang.startsWith("ja")) return { languageCode: "ja-JP", name: "ja-JP-Neural2-B" };
  if (lang.startsWith("zh")) return { languageCode: "cmn-CN", name: "cmn-CN-Wavenet-A" };
  return { languageCode: "en-US", name: "en-US-Neural2-C" };
}
async function synth(text, lang = "en-US") {
  if (!text) return "";
  const voice = pickVoice(lang);
  const [res] = await tts.synthesizeSpeech({
    input: { text },
    voice,
    audioConfig: { audioEncoding: "MP3", speakingRate: 1.0 },
  });
  return res.audioContent?.toString("base64") ?? "";
}

// ---- STT v2 공용 호출
async function sttV2RecognizeBase64(wavB64, hints = [], recognizerOverride) {
  const recognizerPath =
    recognizerOverride ||
    `projects/${PROJECT_ID}/locations/${LOCATION}/recognizers/${RECOGNIZER_ID}`;

  const v2req = {
    recognizer: recognizerPath,
    config: {
      autoDecodingConfig: {},
      features: { enableAutomaticPunctuation: true },
      ...(hints?.length
        ? { adaptation: { phraseSets: [{ phrases: hints.map(v => ({ value: v })), boost: 20.0 }] } }
        : {}),
    },
    content: wavB64,
  };
  const [resp] = await sttV2.recognize(v2req);
  const text = (resp.results || [])
    .map((r) => r.alternatives?.[0]?.transcript || "")
    .join(" ")
    .trim();
  return text;
}

// ── 메인 REST 엔드포인트 ──────────────────────────────────
app.post("/stt/score", upload.single("audio"), async (req, res) => {
  let inPath, outPath;
  try {
    // 0) 업로드 유효성
    if (!req.file) return res.status(400).json({ ok: false, error: "audio file is required" });
    if (!req.file.buffer?.length) return res.status(400).json({ ok: false, error: "empty audio upload" });

    // 1) 입력 파라미터
    const target = req.body?.target || "";
    // v2에서는 Recognizer가 언어를 가짐. lang은 코칭/음성언어용으로 활용
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
    const wantTTS = req.body?.tts === "1"; // ← 응답 음성 포함 여부

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

    // 4) STT: v2 → (옵션) v1 폴백
    const recognizerPath = `projects/${PROJECT_ID}/locations/${LOCATION}/recognizers/${RECOGNIZER_ID}`;
    let transcript = "";

    try {
      transcript = await sttV2RecognizeBase64(wavB64, hints, recognizerPath);
    } catch (e) {
      console.error("[v2 failed]", e?.message || e);
      const USE_V1_FALLBACK = envAny(["USE_V1_FALLBACK"], "0") === "1";
      if (USE_V1_FALLBACK) {
        try {
          const v1req = {
            config: {
              languageCode: langForHints,
              enableAutomaticPunctuation: true,
              ...(hints.length ? { speechContexts: [{ phrases: hints }] } : {}),
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
          throw e1; // 최종 에러
        }
      } else {
        throw e; // 폴백 비활성 시 에러 그대로
      }
    }

    // 5) 스코어링
    const scoring = scoreAttempt({ transcript, targetText: target });

    // 6) 시맨틱(임베딩) — 6초 제한, 실패 시 스킵
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
        const cos = dot / (na * nb);                    // -1..1
        semantic = Math.round(((cos + 1) / 2) * 100);   // 0..100
      } catch (err) {
        console.error("[embedding skipped]", err?.message || err);
      }
    }

    // 7) Gemini 코칭(JSON 강제) — 6초 제한, 실패 시 스킵
    let aiFeedback = null;
    try {
      if (wantCoach) {
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
          gen.generateContent({
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig: { responseMimeType: "application/json" },
          }),
          6000
        );
        const text = r.response?.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
        aiFeedback = JSON.parse(text);
      }
    } catch (err) {
      console.error("[gemini skipped]", err?.message || err);
      aiFeedback = null;
    }

    // 8) (옵션) TTS로 코치 발화 준비
    let coachSpeechB64 = "";
    const speakLine =
      aiFeedback?.next_prompt ||
      (transcript ? "Good job. Tell me more." : "Hello! What would you like to practice?");
    if (wantTTS) {
      try {
        coachSpeechB64 = await synth(speakLine, langForHints);
      } catch (e) {
        console.error("[tts failed]", e?.message || e);
      }
    }

    // 9) 응답
    res.json({ ok: true, transcript, scoring, semantic, aiFeedback, coachSpeechB64, speakLine });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: String(e) });
  } finally {
    try { if (inPath) await fs.unlink(inPath); } catch {}
    try { if (outPath) await fs.unlink(outPath); } catch {}
  }
});

const PORT = process.env.PORT || 8080;
const server = app.listen(PORT, () => console.log("Server on " + PORT));

// ── WS 스트리밍 (pseudo-live) ──────────────────────────────
const wss = new WebSocketServer({ server, path: "/stt/stream" });

// 공용: base64 WAV 를 받아 STT v2로 돌리는 함수 (위에 정의됨)
// async function sttV2RecognizeBase64(...)

wss.on("connection", (ws) => {
  ws.send(JSON.stringify({ type: "ready" }));

  let chunks = [];            // Uint8Array[]
  let timer = null;
  let lastPartial = "";
  let flushing = false;       // 동시 실행 방지
  let totalBytes = 0;
  const MAX_BYTES = 8 * 1024 * 1024; // 8MB 이상이면 오래된 조각 버림

  const flushPartial = async () => {
    if (flushing) return;
    flushing = true;
    try {
      if (!chunks.length) return;
      const inPath = path.join(tmpdir(), `${randomUUID()}.webm`);
      const outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
      const buf = Buffer.concat(chunks);
      await fs.writeFile(inPath, buf);
      await toLinear16(inPath, outPath);
      const wavB64 = (await fs.readFile(outPath)).toString("base64");
      await Promise.allSettled([fs.unlink(inPath), fs.unlink(outPath)]);

      const text = await sttV2RecognizeBase64(wavB64, []);
      if (text && text !== lastPartial) {
        lastPartial = text;
        ws.send(JSON.stringify({ type: "partial", text }));
      }
    } catch (e) {
      // 부분 인식 에러는 조용히 스킵
    } finally {
      flushing = false;
    }
  };

  // 2초마다 부분 인식
  timer = setInterval(flushPartial, 2000);

  ws.on("message", async (data, isBinary) => {
    if (isBinary) {
      const b = Buffer.from(data);
      chunks.push(b);
      totalBytes += b.length;
      while (totalBytes > MAX_BYTES && chunks.length > 1) {
        totalBytes -= chunks[0].length;
        chunks.shift(); // 오래된 조각 제거
      }
      return;
    }
    const msg = data.toString();
    if (msg === "stop") {
      clearInterval(timer);
      try {
        await flushPartial(); // 마지막 부분
        if (chunks.length) {
          const inPath = path.join(tmpdir(), `${randomUUID()}.webm`);
          const outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
          const buf = Buffer.concat(chunks);
          await fs.writeFile(inPath, buf);
          await toLinear16(inPath, outPath);
          const wavB64 = (await fs.readFile(outPath)).toString("base64");
          await Promise.allSettled([fs.unlink(inPath), fs.unlink(outPath)]);

          const finalText = await sttV2RecognizeBase64(wavB64, []);
          ws.send(JSON.stringify({ type: "final", text: finalText }));
        } else {
          ws.send(JSON.stringify({ type: "final", text: "" }));
        }
      } catch (e) {
        ws.send(JSON.stringify({ type: "error", error: e?.message || String(e) }));
      } finally {
        chunks = [];
        lastPartial = "";
        totalBytes = 0;
      }
    }
  });

  ws.on("close", () => {
    clearInterval(timer);
  });
});
