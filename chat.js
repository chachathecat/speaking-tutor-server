import express from 'express';
import fetch from 'node-fetch';

const router = express.Router();

router.post('/chat', async (req,res)=>{
  const { system, history=[], state={}, targetLang='en-US', nativeLang='ko-KR' } = req.body || {};
  const base = process.env.LLM_BASE_URL || 'https://api.openai.com';
  const key  = process.env.LLM_API_KEY;
  const model= process.env.LLM_MODEL || 'gpt-4o-mini';
  if(!key) return res.status(500).json({ ok:false, error:'Missing LLM_API_KEY' });

  const messages = [];
  if (system) messages.push({ role:'system', content: system.replace('{{targetLang}}',targetLang).replace('{{nativeLang}}',nativeLang) });
  for (const m of history) messages.push({ role: m.role==='assistant'?'assistant':'user', content:String(m.content||'') });
  messages.push({ role:'system', content:`Context: ${JSON.stringify({ targetLang, nativeLang, state }).slice(0,1000)}` });

  res.writeHead(200, { 'Content-Type':'text/event-stream; charset=utf-8', 'Cache-Control':'no-cache, no-transform', 'Connection':'keep-alive' });

  try{
    const r = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type':'application/json', 'Authorization': `Bearer ${key}` },
      body: JSON.stringify({ model, stream:true, temperature:0.6, messages })
    });
    if (!r.ok || !r.body) throw new Error(`Upstream ${r.status}`);
    const reader=r.body.getReader(); const dec=new TextDecoder();
    while(true){
      const { value, done } = await reader.read(); if (done) break;
      for (const line of dec.decode(value).split('\n')){
        if (!line.startsWith('data:')) continue;
        const data=line.slice(5).trim(); if(!data || data==='[DONE]') continue;
        try{ const j=JSON.parse(data); const d=j.choices?.[0]?.delta?.content||''; if(d) res.write(`data:${d}\n\n`); }
        catch { res.write(`data:${data}\n\n`); }
      }
    }
  }catch(e){ res.write(`data:[stream error]\n\n`); }
  finally{ res.end(); }
});

export default router;
