// ws-live.js
import { WebSocketServer } from "ws";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";
import { v1 as speechV1 } from "@google-cloud/speech";
import { randomUUID } from "crypto";

if (ffmpegPath) ffmpeg.setFfmpegPath(ffmpegPath);

export function installLiveSTT({ server, path = "/stt/live" }) {
  const stt = new speechV1.SpeechClient({
    keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS
  });

  const wss = new WebSocketServer({ server, path });

  wss.on("connection", (ws) => {
    let ff = null;
    let stream = null;
    let closed = false;

    const config = {
      config: {
        encoding: "LINEAR16",
        sampleRateHertz: 16000,
        languageCode: "en-US",
        enableWordTimeOffsets: true,
        enableWordConfidence: true,
        enableAutomaticPunctuation: true
      },
      interimResults: true,
      singleUtterance: false
    };

    function start() {
      // Opus/mp4 → PCM 파이프
      const pcm = ffmpeg()
        .input("pipe:0")
        .inputOptions(["-thread_queue_size 1024"])
        .outputOptions(["-ac 1", "-ar 16000", "-f s16le", "-acodec pcm_s16le"])
        .on("error", () => {})
        .pipe();

      stream = stt
        .streamingRecognize(config)
        .on("error", (e) => {
          if (!closed) ws.send(JSON.stringify({ type: "error", error: String(e.message || e) }));
        })
        .on("data", (d) => {
          const result = d.results?.[0];
          const alt = result?.alternatives?.[0];
          if (!alt) return;
          const text = alt.transcript || "";
          const words = (alt.words || []).map((w) => ({
            w: w.word,
            start:
              Number(w.startTime?.seconds || 0) +
              Number(w.startTime?.nanos || 0) / 1e9,
            end:
              Number(w.endTime?.seconds || 0) +
              Number(w.endTime?.nanos || 0) / 1e9,
            conf: w.confidence ?? 0
          }));
          ws.send(
            JSON.stringify({
              type: result.isFinal ? "final" : "partial",
              text,
              words
            })
          );
        });

      pcm.on("data", (chunk) => stream.write({ audioContent: chunk }));
      ff = pcm;
    }

    start();
    ws.send(JSON.stringify({ type: "ready", id: randomUUID() }));

    ws.on("message", (data, isBinary) => {
      if (isBinary) {
        ff?.write(data);
        return;
      }
      const s = String(data || "");
      if (s.startsWith("{")) {
        try {
          const j = JSON.parse(s);
          if (j.cmd === "lang" && j.lang) config.config.languageCode = j.lang;
        } catch {}
        return;
      }
      if (s === "stop") {
        try {
          stream?.end();
          ff?.end();
        } catch {}
      }
    });

    ws.on("close", () => {
      closed = true;
      try {
        stream?.end();
      } catch {}
      try {
        ff?.end();
      } catch {}
    });
  });

  return wss;
}
