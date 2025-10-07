import express from 'express';
import Busboy from 'busboy';
import { execa } from 'execa';
import { v2 as speech } from '@google-cloud/speech';

const router = express.Router();
const ffArgs = ['-hide_banner','-loglevel','error','-i','pipe:0','-ac','1','-ar','16000','-f','wav','pipe:1'];

function levenshtein(a,b){ a=(a||'').toLowerCase(); b=(b||'').toLowerCase();
  const m=Array.from({length:a.length+1},()=>Array(b.length+1).fill(0));
  for(let i=0;i<=a.length;i++)m[i][0]=i; for(let j=0;j<=b.length;j++)m[0][j]=j;
  for(let i=1;i<=a.length;i++)for(let j=1;j<=b.length;j++){
    const c=a[i-1]===b[j-1]?0:1; m[i][j]=Math.min(m[i-1][j]+1,m[i][j-1]+1,m[i-1][j-1]+c);
  } return m[a.length][b.length];
}

router.post('/score', async (req,res)=>{
  const bb=Busboy({ headers:req.headers }); const fields={}; let audio=Buffer.alloc(0);
  bb.on('file', (_n,file)=>file.on('data',(d)=>audio=Buffer.concat([audio,d])));
  bb.on('field',(k,v)=>fields[k]=v);
  bb.on('finish', async ()=>{
    try{
      if(!audio.length) return res.status(400).json({ ok:false, error:'audio missing' });
      const { stdout:wav } = await execa('ffmpeg', ffArgs, { encoding:null, input:audio });

      const client = new speech.SpeechClient();
      const lang = fields.lang || 'en-US';
      const recPath = `projects/${process.env.GCP_PROJECT_ID}/locations/${process.env.GCP_LOCATION||'us-central1'}/recognizers/${process.env.GCP_RECOGNIZER_ID}`;
      const [result] = await client.recognize({
        recognizer: recPath,
        config: { autoDecodingConfig:{}, features:{ enableAutomaticPunctuation:true, enableWordTimeOffsets:true }, languageCodes: Array.isArray(lang)?lang:[lang] },
        content: wav.toString('base64')
      });

      const alt = result?.results?.[0]?.alternatives?.[0] || {};
      const text = alt.transcript || '';
      const words = alt.words || [];
      const last  = words.at(-1);
      const dur = Math.max(0.5, last ? Number(last.endOffset?.seconds||0)+(Number(last.endOffset?.nanos||0)/1e9) : 1);

      // 지표 계산
      const wpm = Math.round((text.length/5) / (dur/60));
      const pauses = words.filter((w,i)=> i>0 && ((Number(w.startOffset?.seconds||0)+(Number(w.startOffset?.nanos||0)/1e9)) - (Number(words[i-1].endOffset?.seconds||0)+(Number(words[i-1].endOffset?.nanos||0)/1e9)) > 0.6)).length;
      const meanConf = words.length? (words.reduce((s,w)=>s+Number(w.confidence||0),0)/words.length) : 0.8;
      const fluency = Math.max(30, Math.min(95, 100 - pauses*7 + Math.min(25, (wpm-80)/2)));
      const target = fields.target || '';
      const dist = target ? levenshtein(text, target) : 0;
      const sim = target ? Math.max(0, 100 - Math.round(100*dist/Math.max(1,target.length))) : null;

      const pronunciation = Math.round( (meanConf*100)*0.7 + ((sim ?? 80)*0.3) );
      const prosody = Math.round( Math.min(95, 70 + (wpm-100)/3 - pauses*2) );
      const grammar = 75; // LLM 보조 없이는 보수적 점수(추후 exam.js에서 보완)

      const scoring = {
        total: Math.round( (fluency*0.35) + (pronunciation*0.35) + (prosody*0.20) + (grammar*0.10) ),
        fluency, pronunciation, prosody, grammar
      };

      res.json({
        ok:true,
        transcript:text,
        scoring,
        semantic: sim,
        words:(words||[]).map(w=>({
          w:w.word,
          start:Number(w.startOffset?.seconds||0)+(Number(w.startOffset?.nanos||0)/1e9),
          end:Number(w.endOffset?.seconds||0)+(Number(w.endOffset?.nanos||0)/1e9),
          conf:Number(w.confidence||0)
        }))
      });
    }catch(e){ res.status(500).json({ ok:false, error:String(e.message||e) }); }
  });
  req.pipe(bb);
});

export default router;
