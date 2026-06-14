import {
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";

import { drawMask, loadMaskFromFile } from "./mask.js";
import { setupRecorder } from "./recorder.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: false });

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const maskInput = document.getElementById("maskInput");
const scaleInput = document.getElementById("scaleInput");

let faceLandmarker;
let recorderApi;
let isRunning = false;
let lastVideoTime = -1;
let maskScale = Number(scaleInput.value) / 100;

function setStatus(message) {
  statusEl.textContent = message;
}

function fitCanvasToVideo() {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 }
    },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });

  fitCanvasToVideo();
  recorderApi = setupRecorder(canvas, setStatus);
}

async function initTracker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true
  });
}

function drawCameraFrame() {
  fitCanvasToVideo();

  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function loop() {
  if (!isRunning) return;

  drawCameraFrame();

  if (video.currentTime !== lastVideoTime && faceLandmarker) {
    lastVideoTime = video.currentTime;

    const result = faceLandmarker.detectForVideo(video, performance.now());

    if (result.faceLandmarks?.length) {
      drawMask({
        ctx,
        canvas,
        landmarks: result.faceLandmarks[0],
        blendshapes: result.faceBlendshapes?.[0]?.categories || [],
        matrix: result.facialTransformationMatrixes?.[0],
        maskScale,
        mirrored: true
      });
      setStatus("Face locked — mask active");
    } else {
      setStatus("Searching for face...");
    }
  }

  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", async () => {
  try {
    startBtn.disabled = true;
    setStatus("Starting camera...");

    if (!video.srcObject) {
      await initCamera();
    }

    if (!faceLandmarker) {
      setStatus("Loading face tracker...");
      await initTracker();
    }

    isRunning = true;
    setStatus("Ready — upload a mask image");
    loop();
  } catch (error) {
    console.error(error);
    startBtn.disabled = false;
    setStatus("Camera/tracker failed. Check browser permissions.");
  }
});

maskInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    await loadMaskFromFile(file);
    setStatus(`Mask loaded: ${file.name}`);
  } catch (error) {
    console.error(error);
    setStatus("Could not load that image as a mask.");
  }
});

scaleInput.addEventListener("input", () => {
  maskScale = Number(scaleInput.value) / 100;
});

recordBtn.addEventListener("click", () => {
  if (!recorderApi) {
    setStatus("Start camera before recording.");
    return;
  }

  recorderApi.start();
});

stopBtn.addEventListener("click", () => {
  if (!recorderApi) return;
  recorderApi.stop();
});
