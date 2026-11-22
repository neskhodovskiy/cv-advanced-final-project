'use strict';
const el = id => document.getElementById(id);
const camera = el('camera').getContext('2d');
const pair = el('pairs').getContext('2d');
const map = el('map').getContext('2d');
let scene, terrain, playing = false, timer, generation = 0, displayed = 0;
const acceptedByMode = new Map();

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

function path(ctx, vertices, fill, stroke, width = 1) {
  ctx.beginPath();
  vertices.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
  if (fill) { ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.lineWidth = width; ctx.strokeStyle = stroke; ctx.stroke(); }
}

function circle(ctx, x, y, r, fill, stroke) {
  ctx.beginPath(); ctx.arc(x,y,r,0,2*Math.PI);
  if (fill) { ctx.fillStyle=fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle=stroke; ctx.lineWidth=1.4; ctx.stroke(); }
}

function drawMap(index, accepted) {
  const w=map.canvas.width, h=map.canvas.height, mw=scene.mapWidth, mh=scene.mapHeight, sx=646/(mw+mh), sy=314/(mw+mh), ox=26+mh*sx;
  const project = ([x,y]) => [ox + (x-y)*sx, 34+(x+y)*sy];
  const corners = [[0,0],[mw,0],[mw,mh],[0,mh]].map(project);
  const bg = map.createLinearGradient(0,0,w,h);
  bg.addColorStop(0,'#172c30'); bg.addColorStop(1,'#263f3c');
  map.fillStyle=bg; map.fillRect(0,0,w,h);
  map.save(); map.setTransform(sx,sy,-sx,sy,ox,34);
  for (let t=-1280;t<2150;t+=64) {
    path(map, [[t,-1280],[t,2150]],null,'#39504e',1);
    path(map, [[-1280,t],[2150,t]],null,'#39504e',1);
  }
  map.restore();
  path(map, [corners[3], corners[2], corners[1], [corners[1][0],corners[1][1]+9],
    [corners[2][0],corners[2][1]+9],[corners[3][0],corners[3][1]+9]], '#42574d',null);
  map.save(); map.setTransform(sx,sy,-sx,sy,ox,34);
  map.drawImage(terrain,0,0,mw,mh); map.restore();
  path(map, [...corners,corners[0]],null,'#93a695',1);
  const [cx,cy] = scene.route[index], px=scene.width/2, py=scene.height/2;
  const [l,t,r,b] = scene.rects ? scene.rects[index] : [cx-px,cy-py,cx+px,cy+py];
  const tile = [[l,t],[r,t],[r,b],[l,b]].map(project);
  path(map,tile,'rgba(249,208,112,.18)','#f5d27c',2);
  const route = scene.route.slice(0,index+1).map(project);
  path(map,route,null,'#123c36',6); path(map,route,null,'#a4f2c7',3);
  for (const i of accepted) {
    if (i<=index) { const [x,y] = project(scene.route[i]); circle(map,x,y,2.8,'#ffdd79','#765b2e'); }
  }
  const [x,y] = project([cx,cy]);
  circle(map,x,y,4,'#e3fbea','#446758');
  map.setLineDash([3,3]); path(map, [[x,y-5],[x,y-42]],null,'#d9f4e7',1.6); map.setLineDash([]);
  map.save(); map.translate(x,y-49);
  for (const [ax,ay] of [[-15,-6],[14,-6],[-12,7],[15,6]]) {
    path(map,[[0,0],[ax,ay]],null,'#d6e5e0',3);
    map.beginPath();map.ellipse(ax,ay,7.5,2.8,0,0,2*Math.PI);
    map.fillStyle='#698079';map.fill();map.strokeStyle='#d8eee2';map.lineWidth=1.2;map.stroke();
  }
  path(map,[[-6,1],[-2,-5],[6,-2],[5,4],[-3,5]],'#e4eee4','#344c42',1.2);map.restore();
  map.fillStyle='#bbcfbf'; map.font='14px system-ui'; map.fillText('Схематична висота моделі',16,h-14);
}

function drawPair(q, r, points) {
  const w = scene.width, h = scene.height;
  pair.canvas.width=2*w; pair.canvas.height=h;
  pair.drawImage(q,0,0,w,h); pair.drawImage(r,w,0,w,h);
  const stride=Math.max(1,Math.ceil(points.length/24));
  points.filter((_,i)=>i%stride===0).slice(0,24).forEach(([x,y,u,v])=>{
    path(pair,[[x,y],[w+u,v]],null,'rgba(224,253,155,.7)',.85);
    circle(pair,x,y,2,'#e4ff95'); circle(pair,w+u,v,2,'#e4ff95');
  });
}

function stop(){
  playing=false; clearTimeout(timer); el('play').textContent='▶'; el('play').setAttribute('aria-label','Відтворити');
}

async function refresh() {
  clearTimeout(timer);
  const token=++generation, index=Number(el('seek').value);
  const variant=el('variant').value, method=el('method').value, clahe=Number(el('clahe').checked);
  const recording=el('recording').value;
  const mode=[recording,variant,method,clahe].join('/'), started=performance.now();
  try {
    const qs=new URLSearchParams({recording,i:index,variant,method,clahe});
    const response=await fetch('/api/frame?'+qs);
    if (!response.ok) throw new Error('frame');
    const data=await response.json();
    const [query,reference]=await Promise.all([loadImage(data.query),loadImage(data.reference)]);
    if(token!==generation)return;
    const accepted=acceptedByMode.get(mode)||new Set();
    if(data.inliers>=4)accepted.add(index);
    acceptedByMode.set(mode,accepted);
    displayed=index;
    const cw=camera.canvas.width, ch=camera.canvas.height;
    const scale=Math.min(cw/query.width,ch/query.height), qw=query.width*scale, qh=query.height*scale;
    camera.fillStyle='#172b2a';camera.fillRect(0,0,cw,ch);
    camera.drawImage(query,(cw-qw)/2,(ch-qh)/2,qw,qh);
    drawPair(query,reference,data.points); drawMap(index,accepted);
    el('score').textContent=data.score; el('count').textContent=data.inliers;
    el('latency').textContent=data.algorithm_ms.toFixed(1); el('decode').textContent=data.decode_ms.toFixed(1);
    el('frame-id').textContent='Кадр '+(scene.synthetic ? String(index+1).padStart(2,'0')+' / '+scene.route.length : scene.ids[index]);
    el('time').textContent='00:'+String(Math.floor(index/scene.fps)).padStart(2,'0');
    el('status').className=data.inliers>=4?'':'bad';
    el('status').textContent=data.inliers>=4?'Пару прийнято':'Недостатньо відповідностей';
    if(playing) {
      if(index+1>=scene.route.length)stop();
      else timer=setTimeout(()=>{el('seek').value=index+1;refresh();},Math.max(0,1000/scene.fps-(performance.now()-started)));
    }
  } catch(e) {
    if(token!==generation)return;
    stop();el('status').textContent='Не вдалося прочитати кадр';el('status').className='bad';
  }
}

async function recordingChanged(){
  stop();
  const token=++generation, name=el('recording').value;
  const response=await fetch('/api/scene?recording='+encodeURIComponent(name));
  if(!response.ok)throw new Error('scene');
  const next=await response.json(), image=await loadImage(next.mapUrl);
  if(token!==generation)return;
  scene=next;terrain=image;displayed=0;
  for(const [key,title] of Object.entries(scene.recordings)){
    if(!Array.from(el('recording').options).some(opt=>opt.value===key))
      el('recording').add(new Option(title,key));
  }
  el('seek').max=scene.route.length-1;el('seek').value=0;
  el('note').textContent=(scene.synthetic?'Маршрут і фрагменти карти задані синтетичними анотаціями.':'Маршрут і фрагменти карти відтворено за опублікованими анотаціями UAV-VisLoc. Швидкість відтворення умовна.')+
     ' Висота моделі схематична. Позначки показують прийняті пари серед переглянутих кадрів у вибраному режимі.';
  el('clahe').parentElement.title='CLAHE · clip limit '+scene.clip;
  await refresh();
}

async function init() {
  try{
    el('recording').onchange=()=>recordingChanged().catch(()=>{el('status').textContent='Помилка завантаження запису';});
    el('play').onclick=()=>{
      if(playing){stop();++generation;el('seek').value=displayed;return;}
      playing=true;el('play').textContent='Ⅱ';el('play').setAttribute('aria-label','Пауза');
      if(displayed>=scene.route.length-1)el('seek').value=0;
      refresh();
    };
    el('seek').oninput=()=>{stop();refresh();};
    for(const name of ['method','variant','clahe'])el(name).onchange=()=>{el('seek').value=displayed;refresh();};
    await recordingChanged();
  }catch(e){el('status').textContent='Помилка завантаження сцени';}
}
init();
