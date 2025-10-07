import express from 'express';
import fetch from 'node-fetch';
const router = express.Router();

const base = process.env.LLM_BASE_URL || 'https://api.openai.com';
const key  = process.env.LLM_API_KEY;
const model= process.env.LLM_MODEL || 'gpt-4o-mini';

const SYS = `
You are an expert speaking examiner and coach.
Return strict JSON only. No markdown, no extra text.
If exam=toefl, score Delivery, LanguageUse, TopicDevelopment (0-4) and total (0-30).
If exam=ielts, score FC, LR, GRA, PR bands (0-9) and total (0-9).
If exam=toeic-speak, provide total (0-200) and sub-scores.
Also produce concise coaching:
- "concise_feedback": 3 bullets in {{nativeLang}} (<=14 words each).
- "grammar_fixes": up to 3 objects {before,after,why} (why in {{nativeLang}} <=14 words).
- "pronunciation_tips": up to 3 short strings (phoneme-level if possible).
- "recast": one-sentence corrected version if errors exist (in {{targetLang}}).
Keep tone kind and practical.
`;

router.post('/exam', async (req,res)=>{
  try{
    if(!key) return res.status(500).json({ ok:false, error:'Missing LLM_API_KEY' });
    const { transcript="", words=[], exam="toefl", targetLang="en-US", nativeLang="ko-KR", topic="", level="B1" } = req.body || {};
    const messages = [
      { role:'system', content: SYS.replace('{{nativeLang}}', nativeLang).replace('{{targetLang}}', targetLang) },
      { role:'user', content: JSON.stringify({ transcript, words, exam, topic, level }).slice(0, 12000) }
    ];
    const r = await fetch(`${base}/v1/chat/completions`, {
      method:'POST',
      headers:{ 'Content-Type':'application/json', 'Authorization':`Bearer ${key}` },
      body: JSON.stringify({ model, temperature:0.2, response_format:{ type:'json_object' }, messages })
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error?.message || 'exam upstream');
    const obj = typeof j.choices?.[0]?.message?.content === 'string'
      ? JSON.parse(j.choices[0].message.content) : (j.choices?.[0]?.message?.content || {});
    res.json({ ok:true, rubric: obj.rubric || obj.scores || obj.bands || obj, aiFeedback: {
      concise_feedback: obj.concise_feedback||[],
      grammar_fixes: obj.grammar_fixes||[],
      pronunciation_tips: obj.pronunciation_tips||[],
      recast: obj.recast || ''
    }});
  }catch(e){ res.status(500).json({ ok:false, error:String(e.message||e) }); }
});

export default router;
