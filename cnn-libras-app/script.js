document.getElementById("toggleTheme").addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
});

const IMAGE_X = 64,
  IMAGE_Y = 64;

const LETTERS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'I', 8: 'L', 9: 'M', 10: 'N', 11: 'O', 12: 'P', 13: 'Q',
    14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'V', 19: 'W', 20: 'Y'
};

const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const roiCanvas = document.getElementById("roi");
const roiCtx = roiCanvas.getContext("2d");
const resultEl = document.getElementById("result");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

let model;
let handDetected = false;
let showLandmarks = false;

function resizeOverlayCanvas() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resizeOverlayCanvas();
      resolve(video);
    };
  });
}

async function loadModel() {
  model = await tf.loadLayersModel("tfjs_model/model.json");
  resultEl.innerText = "Modelo carregado!";
}

function drawROIBox() {
  const videoWidth = video.clientWidth;
  const videoHeight = video.clientHeight;
  const roiSize = videoWidth * 0.3;

  const roiX = (videoWidth - roiSize) / 2;
  const roiY = (videoHeight - roiSize) / 2;

  overlayCtx.strokeStyle = handDetected ? "lime" : "red";
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeRect(roiX, roiY, roiSize, roiSize);
}

function predict() {
  drawROIBox();

  if (!handDetected) {
    resultEl.innerText = "Nenhuma mão detectada.";
    return;
  }

  const sx = (video.videoWidth - 196) / 2;
  const sy = (video.videoHeight - 196) / 2;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, IMAGE_X, IMAGE_Y);
  ctx.drawImage(video, sx, sy, 196, 196, 0, 0, IMAGE_X, IMAGE_Y);
  roiCtx.drawImage(video, sx, sy, 196, 196, 0, 0, 240, 240);

  const imageData = ctx.getImageData(0, 0, IMAGE_X, IMAGE_Y);
  const input = tf.browser
    .fromPixels(imageData)
    .toFloat()
    .div(255.0)
    .expandDims();

  model
    .predict(input)
    .array()
    .then((arr) => {
      const probs = arr[0];
      const maxIndex = probs.indexOf(Math.max(...probs));
      const prob = Math.max(...probs) * 100;
      resultEl.innerText = `Letra: ${LETTERS[maxIndex]} (${prob.toFixed(1)}%)`;
    });

  input.dispose();
}

function setupHands() {
  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  hands.onResults((results) => {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    handDetected =
      results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

    drawROIBox();

    if (showLandmarks && results.multiHandLandmarks) {
      overlayCtx.save();
      overlayCtx.scale(-1, 1);
      overlayCtx.translate(-overlay.width, 0);

      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(overlayCtx, landmarks, HAND_CONNECTIONS, {
          color: "cyan",
          lineWidth: 2,
        });
        drawLandmarks(overlayCtx, landmarks, {
          color: "magenta",
          lineWidth: 1,
        });
      }

      overlayCtx.restore();
    }
  });

  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480,
  });

  camera.start();
}

async function init() {
  await setupCamera();
  await loadModel();
  setupHands();
  setInterval(predict, 1000);
  window.addEventListener("resize", resizeOverlayCanvas);
}

document.getElementById("toggleLandmarks").addEventListener("click", () => {
  showLandmarks = !showLandmarks;
  document.getElementById("toggleLandmarks").innerText = showLandmarks
    ? "🙈 Ocultar Landmarks"
    : "🖐️ Mostrar Landmarks";
});

init();
