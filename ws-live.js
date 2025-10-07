import { v2 as speech } from '@google-cloud/speech';
import { PassThrough } from 'node:stream';

const DEFAULT_LANGS = (process.env.DEFAULT_LANGS || 'en-US').split(','); // 다중 가능
const PROJECT = process.env.GCP_PROJECT_ID;
const LOCATION = process.env.GCP_LOCATION || 'us-central1';
const RECOGNIZER_ID = process.env.GCP_RECOGNIZER_ID;

function recognizerPath() {
  if (!PROJECT || !LOCATION || !RECOGNIZER_ID) throw new Error('Missing GCP_* env');
  return `projects/${PROJECT}/locations/${LOCATION}/recognizers/${RECOGNIZER_ID}`;
}

export function setupLiveWS(wss, { logger }) {
  // Heartbeat
  const interval = setInterval(() => {
    wss.clients.forEach((s) => {
      if (!s.isAlive) return s.terminate();
      s.isAlive = false; try{ s.ping(); }catch{}
    });
  }, 15000);
  wss.on('close', ()=>clearInterval(interval));

  wss.on('connection', async (ws) => {
    ws.isAlive = true;
    ws.on('pong', ()=>{ ws.isAlive = true; });

    const client = new speech.SpeechClient();
    let langs = DEFAULT_LANGS; // ['en-US'] 또는 ['en-US','ko-KR', ...]
    let stream = null, audioPipe = null, open = false;

    function startStream(){
      const initial = {
        recognizer: recognizerPath(),
        streamingConfig: {
          config: {
            autoDecodingConfig: {},
            features: {
              enableWordTimeOffsets: true,
              enableWordConfidence: true,
              enableAutomaticPunctuation: true
            },
            languageCodes: langs,  // ★ 여러 언어 동시 허용
            model: 'long'
          },
          interimResults: true
        }
      };
      stream = client.streamingRecognize(); open = true;

      stream.on('data', (data)=>{
        try{
          for (const r of (data.results||[])) {
            const alt = r.alternatives?.[0]; if (!alt) continue;
            const words = (alt.words||[]).map(w=>({
              w: w.word,
              start: Number(w.startOffset?.seconds||0)+(Number(w.startOffset?.nanos||0)/1e9),
              end:   Number(w.endOffset?.seconds||0)+(Number(w.endOffset?.nanos||0)/1e9),
              conf:  Number(w.confidence||0)
            }));
            ws.send(JSON.stringify({ type: r.isFinal?'final':'partial', text: alt.transcript||'', words }));
          }
        }catch(e){ logger.warn({ err:e }, 'WS send failed'); }
      });
      stream.on('error', (err)=>{ logger.error({ err }, 'GCP stream error'); safeClose(); });
      stream.on('end', ()=>{ open=false; });

      audioPipe = new PassThrough();
      audioPipe.on('data', (chunk)=>{ if(open) stream.write({ audio:{ content: chunk } }); });
      stream.write(initial);
    }

    function safeClose(){ try{ stream?.end(); }catch{}; try{ audioPipe?.end(); }catch{}; try{ ws.close(); }catch{}; }

    ws.on('message', (msg, isBinary)=>{
      if (!isBinary){
        try{
          const j = JSON.parse(msg.toString('utf8'));
          if (j.cmd==='lang'){
            // j.lang: string | string[]
            if (Array.isArray(j.lang) && j.lang.length) langs = j.lang;
            else if (typeof j.lang==='string') langs = [ j.lang.includes('-')?j.lang:`${j.lang}-US` ];
          } else if (j.cmd==='startSegment'){ if (!stream) startStream(); }
          else if (j.cmd==='endSegment'){ if (stream){ try{ stream.end(); }catch{}; stream=null; open=false; } }
        }catch{}
        return;
      }
      if (!stream) startStream();
      audioPipe.write(msg); // PCM16LE@16k mono (10–20ms)
    });

    ws.on('close', ()=>safeClose());
    ws.on('error', ()=>safeClose());
  });
}
