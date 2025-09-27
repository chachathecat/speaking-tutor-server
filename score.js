// score.js (drop-in replacement)

export function scoreAttempt({ transcript, targetText, audio = {}, lang = "auto" }) {
  // -------- helpers --------
  const clamp01 = (x) => Math.max(0, Math.min(1, x ?? 0));

  // unicode normalize + toLocaleLowerCase
  const norm = (s, l = undefined) =>
    (s || "").normalize("NFKC").toLocaleLowerCase(l).trim();

  // tiny stopword lists (optional, extend as you like)
  const stop = {
    "en": new Set(["a","an","the","to","of","in","on","at","and","or","is","are","am","was","were","be","been","being","i","you","he","she","it","we","they","that","this","for","with"]),
    "ko": new Set(["은","는","이","가","을","를","에","에서","에게","그리고","또","또는","하지만"]),
    "ja": new Set(["は","が","を","に","へ","と","も","や","の","です","ます","そして"]),
  };

  // tokenization that works across scripts
  function tokenize(text, localeHint = undefined) {
    const t = norm(text, localeHint);
    // Intl.Segmenter falls back gracefully if locale is unknown
    const seg = new Intl.Segmenter(localeHint, { granularity: "word" });
    const out = [];
    for (const { segment, isWordLike } of seg.segment(t)) {
      if (!isWordLike) continue;
      // keep tokens that contain at least one letter/number in ANY script
      if (/\p{L}|\p{N}/u.test(segment)) out.push(segment);
    }
    return out;
  }

  // pick locale for tokenization
  const guessLocale =
    lang === "auto"
      ? guessByText(targetText || transcript)
      : lang;

  function guessByText(s) {
    // very rough heuristic per script
    if (/[가-힣]/.test(s)) return "ko-KR";
    if (/[一-龯ぁ-ゔゞァ-・ヽヾ゛゜ーｱ-ﾝﾞﾟ]/.test(s)) return "ja-JP";
    if (/[一-龥]/.test(s)) return "zh-CN";
    if (/[a-zA-Z]/.test(s)) return "en-US";
    return "en-US";
  }

  const refTokens = tokenize(targetText, guessLocale);
  const hypTokens = tokenize(transcript, guessLocale);

  // optional stopwords removal (lightweight)
  const sw = stop[(guessLocale || "").slice(0,2)] || new Set();
  const refFiltered = refTokens.filter(w => !sw.has(w));
  const hypFiltered = hypTokens.filter(w => !sw.has(w));

  // -------- WER (0~1), guarded --------
  const wer = wordErrorRate(refFiltered, hypFiltered); // 0~1
  const pron_word = clamp01(1 - wer);
  const pron_phon = 0.8 * pron_word + 0.2; // placeholder phoneme weight

  // -------- Fluency --------
  const wpm = Number.isFinite(audio.wpm) ? audio.wpm : 120;
  // logistic around 135 wpm (comfortable), spread=25
  const speedScore = 1 / (1 + Math.exp(-(wpm - 135) / 25));
  // remap logistic (0..1) to centered bell: best ~0.5..0.9
  const speed = clamp01(1 - 2 * Math.abs(speedScore - 0.75)); // peak near 0.75

  const silence = clamp01(audio.silenceRatio ?? 0.15);
  const fluency = clamp01(0.65 * speed + 0.35 * (1 - silence));

  // -------- Prosody --------
  const pitchStd = Number.isFinite(audio.pitchStd) ? audio.pitchStd : 30;
  const pitchMean = Number.isFinite(audio.pitchMean) ? Math.max(1, audio.pitchMean) : 150;
  const varRatio = pitchStd / pitchMean; // variability
  // 0.08~0.22 범위를 괜찮은 억양으로 간주
  const prosody = clamp01((varRatio - 0.06) / 0.16);

  // -------- Grammar (key content coverage) --------
  const keyTokens = refFiltered.filter(w => w.length > 1);
  const miss = keyTokens.filter(w => !hypFiltered.includes(w)).length;
  const coverage = keyTokens.length ? 1 - miss / keyTokens.length : 1;
  const grammar = clamp01(coverage);

  // -------- Weighting --------
  // If no target text (free talk), de-emphasize pronunciation/grammar
  const hasTarget = refTokens.length >= 3; // short prompts won't over-penalize
  const wPron = hasTarget ? 0.40 : 0.25;
  const wFlu  = hasTarget ? 0.30 : 0.45;
  const wPro  = hasTarget ? 0.15 : 0.20;
  const wGrm  = hasTarget ? 0.15 : 0.10;

  const Pron = 0.6 * pron_word + 0.4 * pron_phon;
  const total01 = clamp01(wPron * Pron + wFlu * fluency + wPro * prosody + wGrm * grammar);

  const total = Math.round(total01 * 100);

  // -------- Feedback --------
  const feedback = [];
  if (hasTarget && wer > 0.25) feedback.push("핵심 단어 발음/정확도를 조금 더 또렷하게 해보세요.");
  if (silence > 0.35) feedback.push("말 사이의 정지를 줄이고 문장을 이어서 말해보세요.");
  if (speed < 0.45) feedback.push("말하기 속도를 110~160 WPM 범위로 맞춰보세요.");
  if (hasTarget && grammar < 0.8) feedback.push("질문에 필요한 핵심 단어가 일부 빠졌어요.");

  return {
    total,
    pronunciation: Math.round(clamp01(Pron) * 100),
    fluency: Math.round(fluency * 100),
    prosody: Math.round(prosody * 100),
    grammar: Math.round(grammar * 100),
    feedback,
    meta: {
      lang: guessLocale,
      tokens: { ref: refTokens.length, hyp: hypTokens.length },
      wer: +wer.toFixed(3),
      wpm,
      silence,
      pitchStd,
      pitchMean,
    }
  };
}

// ------- WER on token arrays -------
function wordErrorRate(refArr, hypArr) {
  const n = refArr.length, m = hypArr.length;
  if (!n && !m) return 0;
  if (!n) return 1; // no reference: undefined target → treat as max distance (we'll downweight elsewhere)
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = refArr[i - 1] === hypArr[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[n][m] / Math.max(1, n);
}
