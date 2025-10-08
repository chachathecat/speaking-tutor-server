// chat.js (핵심만)
import fetch from 'node-fetch';

export default async function chatRouter(req, res){
  const { system, history, state, targetLang, nativeLang } = await req.json();

  const systemPrompt = (system || `
You are a warm, upbeat speaking coach and dialogue partner.
Speak concisely (1–2 sentences). Use natural conversational rhythm.
ALWAYS finish with exactly one thoughtful follow-up question.
Correct only impactful mistakes in ≤1 short line.
Use ${targetLang} for conversation; use ${nativeLang} only for a brief tip if necessary.
Adapt topic/level/speed based on state.wpm and state.scoring. Keep it friendly and curious.
`).trim();

  // OpenAI/기타 LLM 엔드포인트로 스트리밍 요청 (모델/키는 ENV)
  const r = await fetch(process.env.LLM_BASE_URL + '/v1/chat/completions', {
    method: 'POST',
    headers: { 
      'Authorization': `Bearer ${process.env.LLM_API_KEY}`,
      'Content-Type': 'application/json',
      // 프런트에서 붙인 X-Edge-Token 검증은 index.js 미들웨어에서 이미 처리했다고 가정
    },
    body: JSON.stringify({
      model: process.env.LLM_MODEL,         // 예: gpt-4o-mini / gpt-4.1-mini 등
      stream: true,
      temperature: 0.7,
      messages: [
        { role:'system', content: systemPrompt },
        ...history.map(m=>({ role:m.role==='assistant'?'assistant':'user', content:m.content })),
        { role:'user', content: `Current state: ${JSON.stringify(state)}` }
      ]
    })
  });

  // SSE 파이프
  res.writeHead(200,{
    'Content-Type':'text/event-stream',
    'Cache-Control':'no-cache',
    'Connection':'keep-alive',
    'Access-Control-Allow-Origin': process.env.CORS_ORIGIN || '*'
  });

  const reader = r.body.getReader();
  const dec = new TextDecoder();
  while(true){
    const { value, done } = await reader.read();
    if(done) break;
    const chunk = dec.decode(value);
    // OpenAI 스트리밍을 "data: {delta}" 형태로 변환
    for(const line of chunk.split('\n')){
      if(!line.trim()) continue;
      try{
        const obj = JSON.parse(line.replace(/^data:\s*/,''));
        const delta = obj.choices?.[0]?.delta?.content || '';
        if(delta) res.write(`data:${delta}\n\n`);
      }catch{ /* noop */ }
    }
  }
  res.write('data:[DONE]\n\n');
  res.end();
}
