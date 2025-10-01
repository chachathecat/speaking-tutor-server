export function scoreAttempt({
  transcript = "",
  targetText = "",
  lang = "auto",             // "en-US" | "ko-KR" | ...
  audio = {}                 // { wpm, ms, silenceRatio, pitchStd, pitchMean }
} = {}) {
  // ---------- helpers ----------
  const clamp01 = (v) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
  const clamp100 = (v) => Math.max(0, Math.min(100, Math.round(v)));
  const norm = (s) => String(s ?? "").normalize("NFKC");

  // 아주 가벼운 불용어 세트(언어별)
  const STOP = {
    en: new Set(["a","an","the","is","are","am","to","of","in","on","for","and","or","with","that","this","it","you","i","me","my","your"]),
    es: new Set(["el","la","los","las","un","una","unos","unas","y","o","de","del","a","en","con","que","es","soy","eres"]),
    ko: new Set(), // 형태소 분석 생략: 빈 세트
    ja: new Set(),
  };

  // 문자로 대략 언어 추정
  function guessByText(s = "") {
    if (/[가-힣]/.test(s)) return "ko-KR";
    if (/[ぁ-ゔゞァ-ヾー一-龯]/.test(s)) return "ja-JP";
    if (/[一-龥]/.test(s)) return "zh-CN";
    if (/[a-zA-Z]/.test(s)) return "en-US";
    return "en-US";
  }

  // Intl.Segmenter 기반 토크나이저(+폴백)
  function tokenize(text, localeHint) {
    const t = norm(text);
    try {
      const seg = new Intl.Segmenter(localeHint, { granularity: "word" });
      const out = [];
      for (const { segment, isWordLike } of seg.segment(t)) {
        if (!isWordLike) continue;
        if (/\p{L}|\p{N}/u.test(segment)) out.push(segment);
      }
      return out;
    } catch {
      return t.split(/[\s.,!?;:()'"、。！？…·【】「」『』—–-]+/u).filter(Boolean);
    }
  }

  // ------- classic WER (Levenshtein / ref length) -------
  function wordErrorRate(refArr, hypArr) {
    const n = refArr.length, m = hypArr.length;
    if (!n && !m) return 0;
    if (!n) return 1;
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

  // ---------- locale & tokens ----------
  const guessLocale = lang === "auto" ? guessByText(targetText || transcript) : lang;
  const langKey = String(guessLocale || "").slice(0, 2).toLowerCase();

  const refTokens = tokenize(targetText, guessLocale);
  const hypTokens = tokenize(transcript, guessLocale);

  const sw = STOP[langKey] || new Set();
  const refFiltered = refTokens.filter((w) => !sw.has(w.toLowerCase()));
  const hypFiltered = hypTokens.filter((w) => !sw.has(w.toLowerCase()));

  const hasTarget = refTokens.length >= 3; // 타겟 문장이 충분히 있을 때만 내용기반 채점 강화

  // ---------- pronunciation(단어 일치 기반) & grammar(커버리지) ----------
  let wer = 1;
  let pron_word = 0.8;   // 프리톡 기본값
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

  // ---------- fluency (속도 + 침묵비율) ----------
  let wpm = Number.isFinite(audio.wpm) ? audio.wpm : 120;
  if (!Number.isFinite(audio.wpm)) {
    if (Number.isFinite(audio.ms) && audio.ms > 0) {
      const words =
        hypTokens.length || Math.max(1, String(transcript).trim().split(/\s+/).length);
      wpm = Math.round(words / (audio.ms / 60000));
    }
  }
  // 135wpm 근처 선호 → 0.75 근처가 최고점이 되도록 매핑
  const speedScore = 1 / (1 + Math.exp(-(wpm - 135) / 25));
  const speed = clamp01(1 - 2 * Math.abs(speedScore - 0.75));

  const silence = clamp01(audio.silenceRatio ?? 0.15);
  const fluency = clamp01(0.65 * speed + 0.35 * (1 - silence));

  // ---------- prosody (피치 변동성) ----------
  const pitchStd = Number.isFinite(audio.pitchStd) ? audio.pitchStd : 30;
  const pitchMean = Number.isFinite(audio.pitchMean) ? Math.max(1, audio.pitchMean) : 150;
  const varRatio = pitchStd / pitchMean;            // 변동성 비율
  // 0.08~0.22 구간을 좋은 범위로 맵핑
  const prosody = clamp01((varRatio - 0.06) / 0.16);

  // ---------- 가중치 ----------
  const wPron = hasTarget ? 0.40 : 0.25;
  const wFlu  = hasTarget ? 0.30 : 0.45;
  const wPro  = hasTarget ? 0.15 : 0.20;
  const wGrm  = hasTarget ? 0.15 : 0.10;

  const Pron = clamp01(0.6 * pron_word + 0.4 * pron_phon);
  const total01 = clamp01(wPron * Pron + wFlu * fluency + wPro * prosody + wGrm * grammar);
  const total = clamp100(total01 * 100);

  // ---------- 피드백 ----------
  const feedback = [];
  if (hasTarget && wer > 0.25) feedback.push("핵심 단어를 또렷하게 말해보자!");
  if (silence > 0.35) feedback.push("말 사이의 쉬는 시간을 조금만 줄여볼까?");
  if (speed < 0.45) feedback.push("조금 더 리듬 있게! 110~160 WPM 속도를 목표로 해봐.");
  if (hasTarget && grammar < 0.8) feedback.push("질문에 필요한 단어가 몇 개 빠졌어.");
  if (hasTarget && missing.length)
    feedback.push(`이 단어들을 꼭 넣어보자: ${missing.slice(0, 3).join(", ")}`);
  while (feedback.length > 4) feedback.pop();

  return {
    total,
    pronunciation: clamp100(Pron * 100),
    fluency: clamp100(fluency * 100),
    prosody: clamp100(prosody * 100),
    grammar: clamp100(grammar * 100),
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
