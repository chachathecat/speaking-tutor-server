import http from 'node:http';
import express from 'express';
import cors from 'cors';
import pino from 'pino';
import pinoHttp from 'pino-http';
import { WebSocketServer } from 'ws';
import crypto from 'node:crypto';

import { setupLiveWS } from './ws-live.js';
import scoreRouter from './score.js';
import chatRouter from './chat.js';
import examRouter from './exam.js';

const PORT = process.env.PORT || 8080;
const ORIGIN = process.env.CORS_ORIGIN || '*';
const EDGE_HMAC_SECRET = process.env.EDGE_HMAC_SECRET || ''; // Worker와 동일한 시크릿

/* ---------------- core app ---------------- */
const app = express();
const logger = pino({ level: process.env.LOG_LEVEL || 'info' });
app.use(pinoHttp({ logger }));
app.use(cors({ origin: ORIGIN, credentials: false }));
app.use(express.json({ limit: '2mb' }));
app.use(express.urlencoded({ extended: false }));

app.get('/healthz', (req,res)=>res.json({ ok:true, ts:Date.now() }));

/* ------------- Edge token verify ------------- */
function b64uToBuf(b64){ return Buffer.from(b64.replace(/-/g,'+').replace(/_/g,'/'), 'base64'); }
function verifyEdgeToken(token, resource){
  if (!EDGE_HMAC_SECRET) return { ok:true, payload:{ dev:true } }; // 개발 중엔 우회 허용
  if (!token || !token.includes('.')) throw new Error('missing token');
  const [b64, sig] = token.split('.', 2);
  const expected = crypto.createHmac('sha256', EDGE_HMAC_SECRET).update(b64).digest();
  const given = b64uToBuf(sig);
  if (expected.length !== given.length || !crypto.timingSafeEqual(expected, given)) throw new Error('bad signature');
  const payload = JSON.parse(Buffer.from(b64, 'base64url').toString('utf8'));
  if (payload.res !== resource) throw new Error('wrong resource');
  if (Date.now() > payload.exp) throw new Error('expired');
  return { ok:true, payload };
}
function edgeGuard(resource){
  return (req,res,next)=>{
    try{
      const token = req.get('x-edge-token') || '';
      if (EDGE_HMAC_SECRET) verifyEdgeToken(token, resource);
      next();
    }catch(e){ res.status(401).json({ ok:false, error:String(e.message||e) }); }
  };
}

/* ---------------- REST routes ---------------- */
app.use('/stt', edgeGuard('stt-rest'), scoreRouter); // /stt/score
app.use('/stt', edgeGuard('stt-rest'), chatRouter);  // /stt/chat (SSE)
app.use('/stt', edgeGuard('stt-rest'), examRouter);  // /stt/exam

/* ---------------- WS upgrade ----------------- */
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer:true, perMessageDeflate:false, maxPayload: 2**20 });
setupLiveWS(wss, { logger });

server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname.startsWith('/stt/live')) {
    try{
      const et = url.searchParams.get('et');
      if (EDGE_HMAC_SECRET) verifyEdgeToken(et, 'stt-live');
    }catch(e){
      socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n'); socket.destroy(); return;
    }
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
  } else {
    socket.destroy();
  }
});

server.listen(PORT, ()=>logger.info({ port:PORT }, 'Server ready'));
