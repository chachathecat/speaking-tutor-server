// index.js
import { WebSocketServer } from "ws";
import express from "express";
import cors from "cors";
import multer from "multer";

import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";

import { promises as fs } from "fs";
import fsSync from "fs";
import { tmpdir } from "os";
import path from "path";
import { randomUUID } from "crypto";

import { v2 as speechV2, v1 as speechV1 } from "@google-cloud/speech";
import textToSpeech from "@google-cloud/text-to-speech";
import { VertexAI } from "@google-cloud/vertexai";
import { GoogleAuth } from "google-auth-library";
import { scoreAttempt } from "./score.js";

/* ===========================
   (옵션) 자격증명/토큰 디버그
   =========================== */
if (process.env.DEBUG_CRED === "1") {
  try {
    const raw = fsSync.readFileSync(process.env.GOOGLE_APPLICATION_CREDENTIALS, "utf8");
    const cred = JSON.parse(raw);
    console.log("[cred]", {
      client_email: cred.client_email,
      private_key_id: cred.private_key_id,
      project_id: cred.project_id,
      hasBeginEnd:
        cred.private_key?.startsWith("-----BEGIN PRIVATE KEY-----\n") &&
        cred.private_key?.trimEnd().endsWith("END PRIVATE KEY-----"),
      hasNewlines: cred.private_key?.includes("\n") ?? false,
    });
  } catch (e) {
    console.error("[cred] READ FAIL:", e);
  }

  (async () => {
    try {
      const auth = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
      const client = await auth.getClient();
      const token = await client.getAccessToken();
      const project = await auth.getProjectId();
      console.log("[auth]", { project, hasToken: !!(token && token.token) });
    } catch (e) {
      console.error("[auth] ADC FAILED:", e);
    }
  })();
}

/* ---------- FFmpeg (Render 등에서 필수) ---------- */
if (ffmpegPath) ffmpeg.setFfmpegPath(ffmpegPath);

/* ---------- Env helpers (과거/현재 키 병행 지원) ---------- */
const envAny = (keys, d = "") => {
  for (const k of keys) if (process.env[k] != null) return process.env[k];
  return d;
};

/* ====== 환경값: 리전 분리 ====== */
const PROJECT_ID = envAny(["GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT_ID", "GCP_PROJECT_ID"], "");

const SPEECH_LOCATION = envAny(
  ["GOOGLE_SPEECH_LOCATION", "SPEECH_LOCATION", "GCP_SPEECH_LOCATION"],
  "global"
);
const VERTEX_LOCATION = envAny(
  ["GEMINI_LOCATION", "VERTEX_LOCATION", "GOOGLE_VERTEX_LOCATION", "GCP_LOCATION", "GOOGLE_LOCATION"],
  "us-central1"
);
const RECOGNIZER_ID = envAny(["GOOGLE_RECOGNIZER_ID", "GCP_RECOGNIZER_ID"], "");
const GEMINI_MODEL = envAny(["GEMINI_MODEL"], "gemini-2.5-flash");

const KEYFILE = process.env.GOOGLE_APPLICATION_CREDENTIALS; // /etc/secrets/gcp.json
const clientOpts = KEYFILE ? { keyFilename: KEYFILE } : {}; // ADC 도 허용

/* ---------- GCP 클라이언트 ---------- */
const vertex = new VertexAI({ project: PROJECT_ID, location: VERTEX_LOCATION });
const sttV2 = new speechV2.SpeechClient(clientOpts);
const sttV1 = new speechV1.SpeechClient(clientOpts);
const tts   = new textToSpeech.TextToSpeechClient(clientOpts);

/* ---------- 작은 유틸 ---------- */
const recognizerPathOf = () =>
  `projects/${PROJECT_ID}/locations/${SPEECH_LOCATION}/recognizers/${RECOGNIZER_ID}`;

const withTimeout = (p, ms) =>
  Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms))]);

/* ---------- Express 기본 ---------- */
const app = express();
app.set("trust proxy", 1);
app.use(cors());
app.use(express.json({ limit: "1mb" })); // 바디가 길어지는 경우 방지
app.options("*", cors());

// 공통 CORS/프록시 헤더 (특히 SSE)
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type,Accept");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// 부팅 로그
console.log("[boot] project:", PROJECT_ID || "(ADC)");
console.log("[boot] speech location:", SPEECH_LOCATION, " recognizer:", RECOGNIZER_ID);
console.log("[boot] vertex location:", VERTEX_LOCATION, " model:", GEMINI_MODEL);

/* --- 대화 세션 메모리 (간단 보관) --- */
const sessions = new Map();
function pushHistory(sid, role, text) {
  if (!sessions.has(sid)) sessions.set(sid, []);
  const hist = sessions.get(sid);
  hist.push({ role, text });
  if (hist.length > 12) hist.splice(0, hist.length - 12);
  return hist;
}

/* --- 레거시 단발 응답 API (유지) --- */
app.post("/chat/reply", async (req, res) => {
  const {
    sessionId = randomUUID(),
    userText = "",
    level = "B1",
    topic = "Free talk",
    lang = "en-US",
  } = req.body || {};

  try {
    if (userText) pushHistory(sessionId, "user", userText);

    const gen = vertex.getGenerativeModel({ model: GEMINI_MODEL });
    const system = `You are an upbeat ESL speaking tutor.
- Keep turns short (max 2 sentences).
- Ask ONE question each turn and drive the conversation.
- Topic: ${topic}, CEFR level: ${level}.
- Speak ${lang}. If the learner struggles, simplify.`;

    const history = (sessions.get(sessionId) || []).flatMap((m) => [
      { role: m.role, parts: [{ text: m.text }] },
    ]);

    const r = await gen.generateContent({
      contents: [{ role: "user", parts: [{ text: system }] }, ...history],
    });

    const text =
      r.response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ||
      "Let's keep going!";

    pushHistory(sessionId, "assistant", text);
    res.json({ ok: true, sessionId, text });
  } catch (e) {
    console.error("[/chat/reply] error:", e);
    res.status(500).json({ ok: false, error: String(e) });
  }
});

/* --- 업로드 메모리 저장 --- */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

/* --- webm/mp4/aac/ogg/mp3 → WAV(PCM_S16LE, mono, 16k) --- */
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

/* ---- TTS ---- */
function pickVoice(lang = "en-US") {
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

/* ---- STT v2 공용 호출 ---- */
async function sttV2RecognizeBase64(wavB64, hints = [], recognizerOverride) {
  const recognizerPath = recognizerOverride || recognizerPathOf();

  const v2req = {
    recognizer: recognizerPath,
    config: {
      autoDecodingConfig: {},
      features: { enableAutomaticPunctuation: true },
      ...(hints?.length
        ? { adaptation: { phraseSets: [{ phrases: hints.map((v) => ({ value: v })), boost: 20.0 }] } }
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

/* ── 파일 업로드 채점 ─────────────────────────────────────── */
app.post("/stt/score", upload.single("audio"), async (req, res) => {
  let inPath, outPath;
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: "audio file is required" });
    if (!req.file.buffer?.length)
      return res.status(400).json({ ok: false, error: "empty audio upload" });

    const target = req.body?.target || "";
    const langForHints = req.body?.lang || "en-US";

    let hints = [];
    try {
      hints = JSON.parse(req.body?.hints || "[]");
      if (!Array.isArray(hints)) hints = [];
    } catch {
      hints = [];
    }
    const wantSemantic = req.body?.semantic === "1";
    const wantCoach    = req.body?.coach === "1";
    const wantTTS      = req.body?.tts === "1";

    const mt = (req.file.mimetype || "").toLowerCase();
    const ext =
      mt.includes("mp4")  ? "mp4"  :
      mt.includes("aac")  ? "aac"  :
      mt.includes("3gpp") ? "3gp"  :
      mt.includes("mpeg") ? "mp3"  :
      mt.includes("ogg")  ? "ogg"  :
      mt.includes("webm") ? "webm" : "dat";

    inPath  = path.join(tmpdir(), `${randomUUID()}.${ext}`);
    outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
    await fs.writeFile(inPath, req.file.buffer);

    await toLinear16(inPath, outPath);
    const wavBytes = await fs.readFile(outPath);
    if (!wavBytes.length) {
      return res.status(400).json({ ok: false, error: "wav conversion produced empty audio" });
    }
    const wavB64 = wavBytes.toString("base64");

    const recognizerPath = recognizerPathOf(); // global 리전 사용
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
          throw e1;
        }
      } else {
        throw e;
      }
    }

    const scoring = scoreAttempt({ transcript, targetText: target });

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
        semantic = Math.round(((cos + 1) / 2) * 100);
      } catch (err) {
        console.error("[embedding skipped]", err?.message || err);
      }
    }

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

    res.json({ ok: true, transcript, scoring, semantic, aiFeedback, coachSpeechB64, speakLine });
  } catch (e) {
    console.error("[/stt/score] error:", e);
    res.status(500).json({ ok: false, error: String(e) });
  } finally {
    try { if (inPath)  await fs.unlink(inPath); } catch {}
    try { if (outPath) await fs.unlink(outPath); } catch {}
  }
});

const PORT = process.env.PORT || 8080;
const server = app.listen(PORT, () => console.log("Server on " + PORT));

/* ---- SSE: /stt/chat (LLM 대화 스트리밍) ---- */
app.post("/stt/chat", async (req, res) => {
  const accept = String(req.get("accept") || "");
  const wantsSSE = accept.includes("text/event-stream");

  const body = req.body || {};
  const {
    system = "",
    history = [], // [{role:'user'|'assistant', content/text:'...'}]
    state = {},   // {topic, level, wpm, scoring, hint, goals}
  } = body;

  // 언어 기본값
  const targetLang = body.targetLang || body.lang || "en-US";
  const nativeLang = body.nativeLang || "ko-KR";

  // 프론트가 개별 필드로 보낸 경우 폴백 병합
  const mergedState = Object.keys(state).length ? state : {
    topic: body.topic, level: body.level, wpm: body.wpm,
    scoring: body.scoring, hint: body.hint, goals: body.goals,
  };

  const gen = vertex.getGenerativeModel({ model: GEMINI_MODEL });

  const sysText = (system || `
You are a friendly speaking coach. Keep replies to 1–2 sentences.
ALWAYS end with exactly one smart follow-up question.
Correct only impactful mistakes in ≤1 short line.
Speak in {{targetLang}}; use {{nativeLang}} only for brief tips.
Adapt topic/level/speed based on 'State'.
  `).replace("{{targetLang}}", targetLang).replace("{{nativeLang}}", nativeLang);

  const toVertexRole = (r) => (r === "assistant" ? "model" : "user");
  const contents = [
    { role: "user", parts: [{ text: sysText }] },
    ...(Array.isArray(history) ? history : []).map(m => ({
      role: toVertexRole(m.role),
      parts: [{ text: String(m.content ?? m.text ?? "") }]
    })),
    { role: "user", parts: [{ text: `State:\n${JSON.stringify(mergedState)}` }] }
  ];

  if (wantsSSE) {
    // 스트리밍 헤더
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-store, must-revalidate",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
      "Access-Control-Allow-Origin": "*",
    });

    const keepalive = setInterval(() => {
      try { res.write(":ka\n\n"); } catch {}
    }, 15000);

    const onClose = () => {
      clearInterval(keepalive);
      try { res.end(); } catch {}
    };
    req.on("close", onClose);
    req.on("aborted", onClose);

    try {
      const stream = await gen.generateContentStream({ contents });
      for await (const chunk of stream.stream) {
        const part = chunk?.candidates?.[0]?.content?.parts?.[0];
        const text = part?.text || "";
        if (text) res.write(`data: ${text}\n\n`);
      }
      res.write("data: [DONE]\n\n");
      res.end();
    } catch (e) {
      res.write(`data: (error) ${String(e?.message || e)}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
    } finally {
      clearInterval(keepalive);
    }
    return;
  }

  // JSON 단발 폴백
  try {
    const r = await gen.generateContent({ contents });
    const text = r.response?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "Let's keep going!";
    res.json({ ok: true, text });
  } catch (e) {
    console.error("[/stt/chat] error:", e);
    res.status(500).json({ ok: false, error: String(e) });
  }
});

/* ───────────────────────── WS 스트리밍 (pseudo-live) ───────────────────────── */

// ★ 첫 바이너리 청크로 컨테이너 추정
function guessExt(buf = Buffer.alloc(0)) {
  try {
    if (buf.length >= 12 && buf.slice(4, 8).toString() === "ftyp") return "mp4"; // ISO-BMFF
    if (buf.slice(0, 4).toString() === "OggS") return "ogg";                      // OGG
    if (buf.slice(0, 4).toString("hex") === "1a45dfa3") return "webm";            // EBML(WebM)
  } catch {}
  return "dat"; // 모르면 ffmpeg 자동탐지
}

const wss = new WebSocketServer({ server, path: "/stt/stream" });

wss.on("connection", (ws) => {
  ws.send(JSON.stringify({ type: "ready" }));

  let chunks = [];
  let firstChunk = null;
  let timer = null;
  let lastPartial = "";
  let flushing = false;
  let totalBytes = 0;
  const MAX_BYTES = 8 * 1024 * 1024;

  const flushPartial = async () => {
    if (flushing) return;
    flushing = true;
    try {
      if (!chunks.length) return;

      const ext = guessExt(firstChunk);
      const inPath = path.join(tmpdir(), `${randomUUID()}.${ext}`);
      const outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
      const buf = Buffer.concat(chunks);
      await fs.writeFile(inPath, buf);
      await toLinear16(inPath, outPath);
      const wavB64 = (await fs.readFile(outPath)).toString("base64");
      await Promise.allSettled([fs.unlink(inPath), fs.unlink(outPath)]);

      const text = await sttV2RecognizeBase64(wavB64, [], recognizerPathOf());
      if (text && text !== lastPartial) {
        lastPartial = text;
        ws.send(JSON.stringify({ type: "partial", text }));
      }
    } catch {
      // 부분 인식 에러는 조용히 스킵
    } finally {
      flushing = false;
    }
  };

  // 1초마다 부분 인식
  timer = setInterval(flushPartial, 1000);

  ws.on("message", async (data, isBinary) => {
    if (isBinary) {
      const b = Buffer.from(data);
      if (!firstChunk) firstChunk = b;
      chunks.push(b);
      totalBytes += b.length;
      while (totalBytes > MAX_BYTES && chunks.length > 1) {
        totalBytes -= chunks[0].length;
        chunks.shift();
      }
      return;
    }

    const msg = data.toString();
    if (msg === "stop") {
      clearInterval(timer);
      try {
        await flushPartial(); // 마지막 부분
        if (chunks.length) {
          const ext = guessExt(firstChunk);
          const inPath = path.join(tmpdir(), `${randomUUID()}.${ext}`);
          const outPath = path.join(tmpdir(), `${randomUUID()}.wav`);
          const buf = Buffer.concat(chunks);
          await fs.writeFile(inPath, buf);
          await toLinear16(inPath, outPath);
          const wavB64 = (await fs.readFile(outPath)).toString("base64");
          await Promise.allSettled([fs.unlink(inPath), fs.unlink(outPath)]);

          const finalText = await sttV2RecognizeBase64(wavB64, [], recognizerPathOf());
          ws.send(JSON.stringify({ type: "final", text: finalText }));
        } else {
          ws.send(JSON.stringify({ type: "final", text: "" }));
        }
      } catch (e) {
        ws.send(JSON.stringify({ type: "error", error: e?.message || String(e) }));
      } finally {
        // 상태 초기화
        chunks = [];
        firstChunk = null;
        lastPartial = "";
        totalBytes = 0;
      }
    }
  });

  ws.on("close", () => clearInterval(timer));
});
