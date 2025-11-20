const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("fileInput");
const consoleEl = document.getElementById("console");

let streaming = false;

function log(msg) {
  consoleEl.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + consoleEl.textContent;
}

async function sendImage(blob) {
  const fd = new FormData();
  fd.append("file", blob);
  const res = await fetch("/v1/infer", { method: "POST", body: fd });
  const json = await res.json();
  drawBoxes(json);
  log("Inference complete");
}

function drawBoxes(data) {
  ctx.lineWidth = 2;
  data.objects.forEach(o => {
    const [x1, y1, x2, y2] = o.box;
    ctx.strokeStyle = "lime";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${o.class} ${(o.score * 100).toFixed(1)}%`;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    const tw = ctx.measureText(label).width + 6;
    ctx.fillRect(x1, y1 - 18, tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 3, y1 - 4);
    if (o.affordances) {
      const aff = o.affordances.map(a => `${a.action}(${(a.p * 100).toFixed(0)}%)`).join(", ");
      ctx.fillText(aff, x1, y2 + 15);
    }
    if (o.anchors) {
      ctx.fillStyle = "gold";
      o.anchors.forEach(([ax, ay]) => {
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  });
}

document.getElementById("startCam").onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.play();
  streaming = true;
  const w = 640, h = 480;
  canvas.width = w;
  canvas.height = h;
  log("Camera started");
  frameLoop();
};

document.getElementById("stopCam").onclick = () => {
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  streaming = false;
  log("Camera stopped");
};

async function frameLoop() {
  if (!streaming) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(sendImage, "image/jpeg");
  requestAnimationFrame(frameLoop);
}

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    canvas.toBlob(sendImage, "image/jpeg");
  };
  img.src = URL.createObjectURL(file);
});
