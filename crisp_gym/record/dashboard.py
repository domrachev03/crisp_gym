"""Web dashboard for dataset recording: FastAPI app + the page it serves.

Pure of ROS: the app talks to a ``monitor`` object with ``snapshot() -> dict``
(latest merged status), ``send_control(cmd) -> bool`` (publish a recording command)
and ``set_gripper(width) -> bool`` (command the gripper). The live ROS monitor that
implements them lives in ``record_dashboard_monitor`` (rclpy); tests inject a fake.
This split keeps the routes + HTML unit-testable without a robot.

Status shape (what ``snapshot`` returns / the page renders)::

    {"recorder": {"connected", "state", "episode", "num_episodes", "fps",
                  "frames", "repo_id", "last_event"},
     "arms": {"right": {"force", "torque", "pos":[x,y,z]}, "left": {...}}}

Layout: the rerun viewer fills the left; episode / loop-rate / arms / controls /
gripper are stacked vertically on the right.
"""

from __future__ import annotations

VALID_COMMANDS = ("record", "save", "delete", "exit")
# Gripper actions -> normalized target width (open = max, close = min), same
# convention as franka_server_standalone (set_target 1.0 / 0.0).
GRIPPER_WIDTHS = {"open": 1.0, "close": 0.0}


def create_app(
    monitor,
    rerun_url: str | None = None,
    rerun_web_port: int = 9090,
    rerun_ws_port: int = 9877,
):
    """Build the FastAPI dashboard app around a status ``monitor``.

    Args:
        monitor: object with ``snapshot() -> dict``, ``send_control(cmd) -> bool``
            and ``set_gripper(width) -> bool``.
        rerun_url: explicit rerun web-viewer URL for the embedded iframe. When ``None``
            the page builds ``http://<host>:<web_port>?url=ws://<host>:<ws_port>`` from
            ``window.location.hostname`` so it works over an ssh tunnel and on the LAN.
        rerun_web_port: rerun web-viewer HTML port (used when ``rerun_url`` is None).
        rerun_ws_port: rerun websocket data port (used when ``rerun_url`` is None).
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="crisp_gym recording dashboard")
    rerun_cfg = {"url": rerun_url, "web_port": rerun_web_port, "ws_port": rerun_ws_port}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/status")
    def status() -> JSONResponse:
        return JSONResponse(monitor.snapshot())

    @app.get("/rerun_config")
    def rerun_config() -> JSONResponse:
        return JSONResponse(rerun_cfg)

    @app.post("/control/{cmd}")
    def control(cmd: str) -> JSONResponse:
        if cmd not in VALID_COMMANDS:
            raise HTTPException(status_code=400, detail=f"unknown command {cmd!r}")
        ok = monitor.send_control(cmd)
        return JSONResponse({"ok": bool(ok), "cmd": cmd})

    @app.post("/gripper/{action}")
    def gripper(action: str) -> JSONResponse:
        if action not in GRIPPER_WIDTHS:
            raise HTTPException(status_code=400, detail=f"unknown gripper action {action!r}")
        ok = monitor.set_gripper(GRIPPER_WIDTHS[action])
        return JSONResponse({"ok": bool(ok), "action": action})

    return app


DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>crisp_gym recording</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:14px 20px;background:#161a22;border-bottom:1px solid #2a2f3a;display:flex;align-items:center;gap:16px}
 h1{font-size:16px;margin:0;font-weight:600}
 .badge{padding:4px 12px;border-radius:14px;font-weight:600;font-size:13px}
 .main{display:flex;gap:16px;padding:16px 20px;align-items:flex-start}
 .left{flex:1.7;min-width:420px}
 .right{flex:1;min-width:300px;max-width:430px;display:flex;flex-direction:column;gap:14px}
 .card{background:#161a22;border:1px solid #2a2f3a;border-radius:10px;padding:16px}
 .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#8a93a6;margin:0 0 12px}
 .big{font-size:30px;font-weight:700}
 .sub{color:#8a93a6;font-size:13px}
 .bar{height:8px;background:#2a2f3a;border-radius:4px;overflow:hidden;margin-top:8px}
 .bar>div{height:100%;background:#4a90d9}
 .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #20242e;font-variant-numeric:tabular-nums}
 .row:last-child{border-bottom:0}
 button{font:inherit;font-weight:600;padding:10px 16px;border-radius:8px;border:1px solid #2a2f3a;background:#222835;color:#e6e6e6;cursor:pointer}
 button:hover{background:#2c3445}
 .ctrls{display:flex;gap:10px;flex-wrap:wrap}
 #go{background:#1f6f3f;border-color:#2a8a50}
 #exit{background:#6f1f1f;border-color:#8a2a2a}
 #gopen{background:#1f4f7a;border-color:#2a6a9a}
 #gclose{background:#7a5b1f;border-color:#9a7a2a}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
 .rerun-head{display:flex;align-items:center;gap:12px;margin-bottom:10px}
 #rerun{width:100%;height:calc(100vh - 132px);min-height:460px;border:1px solid #2a2f3a;border-radius:10px;background:#0b0d11}
 a.rerun-link{color:#7fb2ec;text-decoration:none;font-size:13px}
 a.rerun-link:hover{text-decoration:underline}
</style></head>
<body>
<header>
 <h1>crisp_gym dataset recording</h1>
 <span id="state" class="badge" style="background:#333">…</span>
 <span id="repo" class="sub"></span>
 <span id="conn" class="sub" style="margin-left:auto"></span>
</header>
<div class="main">
 <div class="left">
   <div class="rerun-head">
     <h2 style="font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#8a93a6;margin:0">rerun viewer</h2>
     <a id="rerunLink" class="rerun-link" href="#" target="_blank" rel="noopener">open in new tab ↗</a>
     <span class="sub" id="rerunMsg" style="margin-left:auto"></span>
   </div>
   <iframe id="rerun" src="about:blank" allow="fullscreen"></iframe>
 </div>
 <div class="right">
   <div class="card">
     <h2>Episode</h2>
     <div class="big"><span id="ep">–</span> / <span id="eptot">–</span></div>
     <div class="bar"><div id="epbar" style="width:0%"></div></div>
     <div class="sub" style="margin-top:10px">frames this episode: <span id="frames">–</span></div>
     <div class="sub">last: <span id="event">–</span></div>
   </div>
   <div class="card">
     <h2>Loop rate (target 60)</h2>
     <div class="big"><span id="fps">–</span> <span class="sub">Hz</span></div>
     <div class="bar"><div id="fpsbar" style="width:0%"></div></div>
   </div>
   <div class="card">
     <h2>Arms — force / EE pose</h2>
     <div class="row"><span><span class="dot" id="rdot"></span>right |F|</span><span id="rforce">–</span></div>
     <div class="row"><span>right pos</span><span id="rpos">–</span></div>
     <div class="row"><span><span class="dot" id="ldot"></span>left |F|</span><span id="lforce">–</span></div>
     <div class="row"><span>left pos</span><span id="lpos">–</span></div>
   </div>
   <div class="card">
     <h2>Controls</h2>
     <div class="ctrls">
       <button id="go" onclick="ctl('record')">Start / Stop</button>
       <button onclick="ctl('save')">Save</button>
       <button onclick="ctl('delete')">Delete</button>
       <button id="exit" onclick="ctl('exit')">Exit</button>
     </div>
     <div class="sub" id="ctlmsg" style="margin-top:10px"></div>
   </div>
   <div class="card">
     <h2>Gripper</h2>
     <div class="ctrls">
       <button id="gopen" onclick="grip('open')">Open</button>
       <button id="gclose" onclick="grip('close')">Close</button>
     </div>
     <div class="sub" id="gripmsg" style="margin-top:10px"></div>
   </div>
 </div>
</div>
<script>
const STATE_COLORS={recording:'#1f6f3f',is_waiting:'#7a5b1f',paused:'#1f4f7a',to_be_saved:'#1f6f3f',to_be_deleted:'#6f1f1f',exit:'#444'};
// Rerun web viewer URL: explicit --rerun-url wins, else build from this page's host
// (window.location.hostname) so it works over an ssh tunnel AND on the LAN. Embedded
// inline (always visible): set the iframe src on load.
let RERUN_URL=null;
async function loadRerunConfig(){
  try{
    const c=await (await fetch('/rerun_config')).json();
    const host=window.location.hostname||'127.0.0.1';
    RERUN_URL=c.url||('http://'+host+':'+c.web_port+'?url=ws://'+host+':'+c.ws_port);
    document.getElementById('rerunLink').href=RERUN_URL;
    document.getElementById('rerun').src=RERUN_URL;
  }catch(e){document.getElementById('rerunMsg').textContent='rerun config error';}
}
loadRerunConfig();
function fmt(v,d=2){return (v===undefined||v===null)?'–':Number(v).toFixed(d);}
function pos(p){return p?('['+p.map(x=>x.toFixed(2)).join(', ')+']'):'–';}
async function ctl(cmd){
  const m=document.getElementById('ctlmsg');
  try{const r=await fetch('/control/'+cmd,{method:'POST'});const j=await r.json();
    m.textContent=(j.ok?'sent: ':'failed: ')+cmd;}catch(e){m.textContent='error: '+e;}
}
async function grip(action){
  const m=document.getElementById('gripmsg');
  try{const r=await fetch('/gripper/'+action,{method:'POST'});const j=await r.json();
    m.textContent=(j.ok?'sent: ':'unavailable: ')+action;}catch(e){m.textContent='error: '+e;}
}
async function tick(){
  try{
    const s=await (await fetch('/status')).json();
    const rec=s.recorder||{}, arms=s.arms||{};
    const conn=rec.connected;
    document.getElementById('conn').textContent=conn?'● recorder connected':'○ no recorder';
    document.getElementById('conn').style.color=conn?'#4a9':'#a55';
    const st=rec.state||'—';
    const b=document.getElementById('state'); b.textContent=st; b.style.background=STATE_COLORS[st]||'#333';
    document.getElementById('repo').textContent=rec.repo_id||'';
    document.getElementById('ep').textContent=rec.episode??'–';
    document.getElementById('eptot').textContent=rec.num_episodes??'–';
    document.getElementById('frames').textContent=rec.frames??'–';
    document.getElementById('event').textContent=rec.last_event||'–';
    const ept=(rec.num_episodes? (100*(rec.episode||0)/rec.num_episodes):0);
    document.getElementById('epbar').style.width=ept+'%';
    const fps=rec.fps||0;
    document.getElementById('fps').textContent=fmt(fps,1);
    const fb=document.getElementById('fpsbar');
    fb.style.width=Math.min(100,100*fps/60)+'%';
    fb.style.background=(fps>=55?'#1f6f3f':fps>=45?'#7a5b1f':'#6f1f1f');
    for(const side of ['right','left']){
      const a=arms[side]||{};
      document.getElementById(side[0]+'force').textContent=fmt(a.force)+' N';
      document.getElementById(side[0]+'pos').textContent=pos(a.pos);
      const d=document.getElementById(side[0]+'dot');
      d.style.background=(a.force>3?'#d9734a':'#4a9');
    }
  }catch(e){document.getElementById('conn').textContent='○ dashboard error';}
}
setInterval(tick,200); tick();
</script>
</body></html>
"""
