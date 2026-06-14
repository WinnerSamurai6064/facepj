function pickMimeType() {
  const options = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4"
  ];

  return options.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

export function setupRecorder(canvas, setStatus = () => {}) {
  let recorder = null;
  let chunks = [];

  function start() {
    if (!window.MediaRecorder) {
      setStatus("This browser does not support MediaRecorder.");
      return;
    }

    if (recorder && recorder.state === "recording") {
      setStatus("Already recording...");
      return;
    }

    chunks = [];
    const stream = canvas.captureStream(30);
    const mimeType = pickMimeType();

    recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

    recorder.ondataavailable = (event) => {
      if (event.data?.size > 0) chunks.push(event.data);
    };

    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      const blob = new Blob(chunks, { type });
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = `facepj-${Date.now()}.${ext}`;
      document.body.appendChild(link);
      link.click();
      link.remove();

      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setStatus("Recording saved locally");
    };

    recorder.start(250);
    setStatus("Recording...");
  }

  function stop() {
    if (!recorder || recorder.state === "inactive") {
      setStatus("No active recording.");
      return;
    }

    recorder.stop();
  }

  return { start, stop };
}
