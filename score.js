// score.js — robust, kid-friendly scoring with fallbacks
// Usage: import { scoreAttempt } from "./score.js";

// score.js
export function scoreAttempt({ transcript = "", targetText = "" }) {
  const t = String(transcript || "").trim();
  const ref = String(targetText || "").trim();

  const len = t.split(/\s+/).filter(Boolean).length;
  const punct = (t.match(/[.?!,]/g) || []).length;
  const longWords = (t.match(/\b\w{7,}\b/g) || []).length;
  const hasFillers = /\b(uh|um|like|you know|well)\b/i.test(t);

  // 어림 지표
  let pron = 70 + Math.min(20, longWords * 2) - (hasFillers ? 5 : 0);
  let flu = 50 + Math.min(35, Math.max(0, len - 8) * 3) + Math.min(10, punct * 2);
  let pro = 55 + Math.min(30, punct * 5);
  let gra = 65 + Math.min(30, longWords * 3) - (/[?!,]$/.test(t) ? 0 : 3);

  if (!t) pron = flu = pro = gra = 0;

  const clamp = (v) => Math.max(0, Math.min(100, Math.round(v)));
  pron = clamp(pron); flu = clamp(flu); pro = clamp(pro); gra = clamp(gra);

  const total = clamp(0.25 * pron + 0.25 * flu + 0.25 * pro + 0.25 * gra);

  const feedback = [];
  if (flu < 55) feedback.push("조금 더 자연스럽게 이어 말해보세요(연음/리듬).");
  if (gra < 60) feedback.push("짧은 문장부터 정확하게—주어/동사 빠짐 체크.");
  if (pro < 60) feedback.push("문장 끝 억양을 낮추고 쉼표로 리듬을 만들기.");
  if (pron < 60) feedback.push("강세(syllable stress)와 모음 길이에 집중해보세요.");

  return { total, pronunciation: pron, fluency: flu, prosody: pro, grammar: gra, feedback };
}


  // very rough script guess
  function guessByText(s = "") {
    if (/[가-힣]/.test(s)) return "ko-KR";
    if (/[ぁ-ゔゞァ-ヾー一-龯]/.test(s)) return "ja-JP";
    if (/[一-龥]/.test(s)) return "zh-CN";
    if (/[a-zA-Z]/.test(s)) return "en-US";
    return "en-US";
  }

  // tokenizer with Intl.Segmenter + safe fallback
  function tokenize(text, localeHint) {
    const t = norm(text, localeHint);
    try {
      const seg = new Intl.Segmenter(localeHint, { granularity: "word" });
      const out = [];
      for (const { segment, isWordLike } of seg.segment(t)) {
        if (!isWordLike) continue;
        if (/\p{L}|\p{N}/u.test(segment)) out.push(segment);
      }
      return out;
    } catch {
      // fallback: split on whitespace & common punctuation
      return t.split(/[\s.,!?;:()'"、。！？…·【】「」『』—–-]+/u).filter(Boolean);
    }
  }

  // ---------- locale & tokens ----------
  const guessLocale = lang === "auto" ? guessByText(targetText || transcript) : lang;
  const langKey = (guessLocale || "").slice(0, 2);
  const refTokens = tokenize(targetText, guessLocale);
  const hypTokens = tokenize(transcript, guessLocale);

  // light stopword removal (keeps content words)
  const sw = stop[langKey] || new Set();
  const refFiltered = refTokens.filter((w) => !sw.has(w));
  const hypFiltered = hypTokens.filter((w) => !sw.has(w));

  // hasTarget: only if we have a meaningful reference (avoid over-penalty in free talk)
  const hasTarget = refTokens.length >= 3;

  // ---------- pronunciation (WER proxy) & grammar (coverage) ----------
  let wer = 1;
  let pron_word = 0.8;   // neutral defaults for free talk
  let pron_phon = 0.85;
  let grammar = 1;
  let missing = [];

  if (hasTarget) {
    wer = wordErrorRate(refFiltered, hypFiltered);  // 0..1
    pron_word = clamp01(1 - wer);
    pron_phon = clamp01(0.8 * pron_word + 0.2);

    const keyTokens = refFiltered.filter((w) => w.length > 1);
    missing = keyTokens.filter((w) => !hypFiltered.includes(w));
    const miss = missing.length;
    grammar = clamp01(keyTokens.length ? 1 - miss / keyTokens.length : 1);
  }

  // ---------- fluency (speed + silence) ----------
  // WPM: prefer provided, else compute from audio.ms or token count
  let wpm = Number.isFinite(audio.wpm) ? audio.wpm : 120;
  if (!Number.isFinite(audio.wpm)) {
    if (Number.isFinite(audio.ms) && audio.ms > 0) {
      const words = hypTokens.length || Math.max(1, (transcript || "").trim().split(/\s+/).length);
      wpm = Math.round(words / (audio.ms / 60000));
    }
  }

  // logistic preference around ~135 wpm, convert to bell peaking near 0.75
  const speedScore = 1 / (1 + Math.exp(-(wpm - 135) / 25));
  const speed = clamp01(1 - 2 * Math.abs(speedScore - 0.75));

  const silence = clamp01(audio.silenceRatio ?? 0.15);
  const fluency = clamp01(0.65 * speed + 0.35 * (1 - silence));

  // ---------- prosody (simple pitch variability proxy) ----------
  const pitchStd = Number.isFinite(audio.pitchStd) ? audio.pitchStd : 30;
  const pitchMean = Number.isFinite(audio.pitchMean) ? Math.max(1, audio.pitchMean) : 150;
  const varRatio = pitchStd / pitchMean;          // variability ratio
  // treat 0.08~0.22 as “good” range → map to 0..1
  const prosody = clamp01((varRatio - 0.06) / 0.16);

  // ---------- weights ----------
  const wPron = hasTarget ? 0.40 : 0.25;
  const wFlu  = hasTarget ? 0.30 : 0.45;
  const wPro  = hasTarget ? 0.15 : 0.20;
  const wGrm  = hasTarget ? 0.15 : 0.10;

  const Pron = clamp01(0.6 * pron_word + 0.4 * pron_phon);
  const total01 = clamp01(wPron * Pron + wFlu * fluency + wPro * prosody + wGrm * grammar);
  const total = Math.round(total01 * 100);

  // ---------- feedback (kid-friendly, concise) ----------
  const feedback = [];
  if (hasTarget && wer > 0.25) feedback.push("핵심 단어를 또렷하게 말해보자!");
  if (silence > 0.35) feedback.push("말 사이의 쉬는 시간을 조금만 줄여볼까?");
  if (speed < 0.45) feedback.push("조금 더 리듬 있게! 110~160 WPM 속도를 목표로 해봐.");
  if (hasTarget && grammar < 0.8) feedback.push("질문에 필요한 단어가 몇 개 빠졌어.");
  if (hasTarget && missing.length)
    feedback.push(`이 단어들을 꼭 넣어보자: ${missing.slice(0, 3).join(", ")}`);

  // cap to 4 bullets
  while (feedback.length > 4) feedback.pop();

  return {
    total,
    pronunciation: Math.round(Pron * 100),
    fluency: Math.round(fluency * 100),
    prosody: Math.round(prosody * 100),
    grammar: Math.round(grammar * 100),
    feedback,
    meta: {
      lang: guessLocale,
      tokens: { ref: refTokens.length, hyp: hypTokens.length },
      wer: hasTarget ? +wer.toFixed(3) : null,
      wpm,
      silence,
      pitchStd,
      pitchMean,
      missing: hasTarget ? missing.slice(0, 5) : [],
    },
  };
}

// ------- classic WER on token arrays (Levenshtein distance / ref length) -------
function wordErrorRate(refArr, hypArr) {
  const n = refArr.length, m = hypArr.length;
  if (!n && !m) return 0;
  if (!n) return 1; // no reference → undefined target; caller down-weights elsewhere

  // DP table
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = refArr[i - 1] === hypArr[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,          // deletion
        dp[i][j - 1] + 1,          // insertion
        dp[i - 1][j - 1] + cost    // substitution
      );
    }
  }
  return dp[n][m] / Math.max(1, n);
}
