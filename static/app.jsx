const { useEffect, useRef, useState } = React;

function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const requestInFlightRef = useRef(false);

  const [status, setStatus] = useState('Opening camera...');
  const [emotion, setEmotion] = useState('Waiting for detection');
  const [confidence, setConfidence] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [error, setError] = useState('');

  const clearOverlay = () => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const drawOverlay = (result) => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!result.detected || !result.box || !result.frame) return;

    const scaleX = canvas.width / result.frame.width;
    const scaleY = canvas.height / result.frame.height;
    const { x, y, w, h } = result.box;

    ctx.strokeStyle = '#31e981';
    ctx.lineWidth = 3;
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);

    const label = `${result.label} ${(result.confidence * 100).toFixed(1)}%`;
    ctx.font = '600 16px Arial';
    ctx.fillStyle = '#31e981';
    ctx.fillRect(x * scaleX, Math.max(0, y * scaleY - 28), Math.max(140, label.length * 9), 24);
    ctx.fillStyle = '#062316';
    ctx.fillText(label, x * scaleX + 8, Math.max(17, y * scaleY - 10));
  };

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const stopCamera = () => {
    stopPolling();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraOn(false);
    setStatus('Camera stopped.');
    clearOverlay();
  };

  const sendFrame = async () => {
    const video = videoRef.current;
    const capture = captureRef.current;

    if (!video || !capture || requestInFlightRef.current) return;
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;

    requestInFlightRef.current = true;

    try {
      capture.width = video.videoWidth;
      capture.height = video.videoHeight;
      const ctx = capture.getContext('2d');
      ctx.drawImage(video, 0, 0, capture.width, capture.height);

      const image = capture.toDataURL('image/jpeg', 0.85);
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image }),
      });

      if (!response.ok) {
        throw new Error('Prediction request failed.');
      }

      const result = await response.json();
      drawOverlay(result);

      if (result.detected) {
        setEmotion(result.label);
        setConfidence((result.confidence * 100).toFixed(2));
        setStatus('Live detection running.');
        setError('');
      } else {
        setEmotion('No face detected');
        setConfidence(null);
        setStatus(result.message || 'No face detected.');
        setError('');
        clearOverlay();
      }
    } catch (err) {
      setError(err.message || 'Prediction failed.');
      setStatus('Prediction paused.');
    } finally {
      requestInFlightRef.current = false;
    }
  };

  const startPolling = () => {
    stopPolling();
    intervalRef.current = setInterval(sendFrame, 900);
  };

  const startCamera = async () => {
    try {
      setError('');
      setStatus('Requesting camera permission...');

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 960 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraOn(true);
      setStatus('Camera ready. Detecting emotion...');
      startPolling();
      setTimeout(sendFrame, 300);
    } catch (err) {
      setError('Camera access was blocked or is unavailable in this browser.');
      setStatus('Click Start Camera to try again after allowing permission.');
      setIsCameraOn(false);
    }
  };

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Emotion Detection CNN</p>
          <h1>Live facial emotion detection from your browser camera</h1>
          <p className="lead">
            This deployed version keeps your trained model on the server and uses the browser camera in this tab to capture frames for live emotion prediction.
          </p>
          <div className="stats-row">
            <div className="stat-box">
              <span className="stat-label">Best validation accuracy</span>
              <strong>61.75%</strong>
            </div>
            <div className="stat-box">
              <span className="stat-label">Classes</span>
              <strong>7 emotions</strong>
            </div>
          </div>
          <div className="controls-row">
            <button className="primary-btn" onClick={startCamera}>Start Camera</button>
            <button className="ghost-btn" onClick={stopCamera}>Stop</button>
          </div>
          <p className="status-text">{status}</p>
          {error ? <p className="error-text">{error}</p> : null}
        </div>

        <div className="camera-card">
          <div className="video-wrap">
            <video ref={videoRef} className="camera-feed" muted playsInline autoPlay />
            <canvas ref={overlayRef} className="overlay-canvas" />
            <canvas ref={captureRef} className="capture-canvas" />
          </div>

          <div className="result-bar">
            <div>
              <span className="result-label">Detected emotion</span>
              <strong>{emotion}</strong>
            </div>
            <div>
              <span className="result-label">Confidence</span>
              <strong>{confidence ? `${confidence}%` : '--'}</strong>
            </div>
            <div>
              <span className="result-label">Camera</span>
              <strong>{isCameraOn ? 'Live' : 'Off'}</strong>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
