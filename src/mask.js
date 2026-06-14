let activeMask = new Image();
let hasUserMask = false;

function makeFallbackMask() {
  const fallback = document.createElement("canvas");
  fallback.width = 512;
  fallback.height = 640;

  const fctx = fallback.getContext("2d");
  const grad = fctx.createLinearGradient(0, 0, 512, 640);
  grad.addColorStop(0, "rgba(255, 160, 64, 0.92)");
  grad.addColorStop(1, "rgba(90, 36, 12, 0.92)");

  fctx.fillStyle = grad;
  fctx.beginPath();
  fctx.ellipse(256, 320, 196, 265, 0, 0, Math.PI * 2);
  fctx.fill();

  fctx.fillStyle = "rgba(0, 0, 0, 0.32)";
  fctx.beginPath();
  fctx.ellipse(190, 265, 28, 18, 0, 0, Math.PI * 2);
  fctx.ellipse(322, 265, 28, 18, 0, 0, Math.PI * 2);
  fctx.fill();

  fctx.fillStyle = "rgba(0, 0, 0, 0.42)";
  fctx.beginPath();
  fctx.ellipse(256, 385, 48, 16, 0, 0, Math.PI * 2);
  fctx.fill();

  activeMask.src = fallback.toDataURL("image/png");
}

makeFallbackMask();

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function point(landmark, canvas, mirrored) {
  const x = mirrored ? 1 - landmark.x : landmark.x;
  return {
    x: x * canvas.width,
    y: landmark.y * canvas.height
  };
}

function blendScore(blendshapes, name) {
  return blendshapes.find((shape) => shape.categoryName === name)?.score || 0;
}

function matrixRotationZ(matrix) {
  if (!matrix?.data) return null;

  const m = matrix.data;
  return Math.atan2(m[1], m[0]);
}

export function loadMaskFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      activeMask = img;
      hasUserMask = true;
      URL.revokeObjectURL(url);
      resolve();
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to decode mask image"));
    };

    img.src = url;
  });
}

export function drawMask({
  ctx,
  canvas,
  landmarks,
  blendshapes,
  matrix,
  maskScale = 1.32,
  mirrored = false
}) {
  const forehead = point(landmarks[10], canvas, mirrored);
  const chin = point(landmarks[152], canvas, mirrored);
  const leftSide = point(landmarks[234], canvas, mirrored);
  const rightSide = point(landmarks[454], canvas, mirrored);
  const nose = point(landmarks[1], canvas, mirrored);

  const upperLip = landmarks[13];
  const lowerLip = landmarks[14];

  const faceWidth = Math.abs(rightSide.x - leftSide.x) * maskScale;
  const faceHeight = Math.abs(chin.y - forehead.y) * maskScale * 1.18;

  const rawRotation = matrixRotationZ(matrix);
  const sideRotation = Math.atan2(rightSide.y - leftSide.y, rightSide.x - leftSide.x);
  const rotation = mirrored ? -1 * (rawRotation ?? sideRotation) : rawRotation ?? sideRotation;

  const jawOpen = blendScore(blendshapes, "jawOpen");
  const smileLeft = blendScore(blendshapes, "mouthSmileLeft");
  const smileRight = blendScore(blendshapes, "mouthSmileRight");
  const cheekLeft = blendScore(blendshapes, "cheekSquintLeft");
  const cheekRight = blendScore(blendshapes, "cheekSquintRight");

  const mouthOpen = clamp(distance(upperLip, lowerLip) * 13 + jawOpen, 0, 1.2);
  const smile = clamp((smileLeft + smileRight) * 0.5, 0, 1);
  const cheek = clamp((cheekLeft + cheekRight) * 0.5, 0, 1);

  ctx.save();
  ctx.translate(nose.x, nose.y + faceHeight * 0.03);
  ctx.rotate(rotation);

  // Cheap deformation illusion: good enough for the hackathon MVP.
  ctx.scale(1 + smile * 0.045 + cheek * 0.035, 1 + mouthOpen * 0.035);

  ctx.globalAlpha = hasUserMask ? 0.93 : 0.74;

  if (activeMask.complete && activeMask.naturalWidth > 0) {
    ctx.drawImage(activeMask, -faceWidth / 2, -faceHeight / 2, faceWidth, faceHeight);
  }

  // Live mouth cue. This keeps static PNG masks feeling alive.
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = "rgba(0, 0, 0, 0.62)";
  ctx.beginPath();
  ctx.ellipse(
    0,
    faceHeight * 0.18,
    faceWidth * (0.075 + smile * 0.035),
    faceHeight * (0.012 + mouthOpen * 0.055),
    0,
    0,
    Math.PI * 2
  );
  ctx.fill();

  ctx.restore();
}
