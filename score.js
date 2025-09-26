export function scoreAttempt({ transcript, targetText, audio = {} }) {
  const clean = s => (s || '').toLowerCase().replace(/[^a-z'\s]/g, ' ').replace(/\s+/g, ' ').trim();
  const ref = clean(targetText);
  const hyp = clean(transcript);

  const wer = wordErrorRate(ref.split(' ').filter(Boolean), hyp.split(' ').filter(Boolean)); // 0~1
  const pron_word = Math.max(0, 1 - wer);
  const pron_phon = 0.8 * pron_word + 0.2; // v0 간이

  const wpm = audio.wpm ?? 120;
  const speedScore = Math.max(0, 1 - Math.abs(wpm - 130) / 80);
  const silence = Math.min(1, audio.silenceRatio ?? 0.15);
  const fluency = 0.6 * speedScore + 0.4 * (1 - silence);

  const pitchStd = audio.pitchStd ?? 30, pitchMean = audio.pitchMean ?? 150;
  const varRatio = pitchStd / Math.max(1, pitchMean);
  const prosody = Math.max(0, Math.min(1, (varRatio - 0.05) / 0.15));

  const keyTokens = ref.split(' ').filter(w => w.length > 2);
  const miss = keyTokens.filter(w => !hyp.includes(w)).length;
  const grammar = Math.max(0, 1 - miss / Math.max(1, keyTokens.length));

  const Pron = 0.6 * pron_word + 0.4 * pron_phon;
  const total = Math.round(100 * (0.40 * Pron + 0.30 * fluency + 0.15 * prosody + 0.15 * grammar));
  const feedback = [];
  if (wer > 0.2) feedback.push("발음 정확도를 조금만 더! 핵심 단어를 또렷하게.");
  if (silence > 0.3) feedback.push("말 사이 간격을 줄여서 더 자연스럽게 이어 말해요.");
  if (speedScore < 0.5) feedback.push("속도를 110~160 WPM 사이로 맞춰 보세요.");
  if (grammar < 0.8) feedback.push("핵심 단어가 빠졌어요. 문장을 다시 확인!");

  return {
    total,
    pronunciation: Math.round(Pron * 100),
    fluency: Math.round(fluency * 100),
    prosody: Math.round(prosody * 100),
    grammar: Math.round(grammar * 100),
    feedback
  };
}

function wordErrorRate(refArr, hypArr) {
  const n = refArr.length, m = hypArr.length;
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = (refArr[i - 1] === hypArr[j - 1]) ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return n ? dp[n][m] / n : 0;
}
