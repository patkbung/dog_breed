'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Dialog, DialogPanel } from '@headlessui/react'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import * as tf from '@tensorflow/tfjs'
import Marquee from 'react-fast-marquee'


// ── Hook โหลดโมเดล ─────────────────────────────────────────────────────────
function useDogModel() {
  const modelRef = useRef(null)
  const [classNames, setClassNames] = useState([])
  const [ready, setReady] = useState(false)
  const [loadingProgress, setLoadingProgress] = useState(0)

  useEffect(() => {
    async function load() {
      const res = await fetch('/tfjs_graph_model_v2/class_names.json')
      const names = await res.json()
      setClassNames(names)
      setLoadingProgress(30)

      const model = await tf.loadGraphModel('/tfjs_graph_model_v2/model.json', {
        onProgress: (fraction) => {
          setLoadingProgress(30 + Math.round(fraction * 70))
        },
      })
      modelRef.current = model

      // warm up
      const dummy = tf.zeros([1, 224, 224, 3])
      model.predict(dummy).dispose()
      dummy.dispose()

      setReady(true)
      setLoadingProgress(100)
    }
    load()
  }, [])

  const predict = async (imageElement) => {
    if (!modelRef.current || !classNames.length) return null

    return tf.tidy(() => {
      const tensor = tf.browser
        .fromPixels(imageElement)
        .resizeBilinear([224, 224])
        .toFloat()
        .expandDims(0)

      const predictions = modelRef.current.predict(tensor)
      // graph model อาจ return object ให้ดึง values ออกมาแบบนี้
      const values = (predictions.dataSync ? predictions : Object.values(predictions)[0]).dataSync()

      const top3 = Array.from(values)
        .map((confidence, i) => ({
          breed: classNames[i].replace(/_/g, ' '),
          confidence,
        }))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3)

      return { breed: top3[0].breed, confidence: top3[0].confidence, top3 }
    })
  }

  return { ready, predict, loadingProgress }
}

// ── Camera / Upload Drop Zone ──────────────────────────────────────────────
function DogImageCapture({ onImageReady }) {
  const [mode, setMode] = useState(null)
  const [preview, setPreview] = useState(null)
  const [facingMode, setFacingMode] = useState('environment')
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices().then((devices) => {
      const cameras = devices.filter((d) => d.kind === 'videoinput')
      setHasMultipleCameras(cameras.length > 1)
    })
  }, [])

  const startCamera = useCallback(async (facing = facingMode) => {
    setError(null)
    try {
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop())
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
    } catch {
      setError('ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้งานกล้อง')
    }
  }, [facingMode])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }, [])

  useEffect(() => () => stopCamera(), [stopCamera])

  const switchCamera = () => {
    const next = facingMode === 'environment' ? 'user' : 'environment'
    setFacingMode(next)
    startCamera(next)
  }

  const capturePhoto = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92)
    setPreview(dataUrl)
    stopCamera()
    setMode('preview')
    onImageReady?.(dataUrl)
  }

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    setMode('preview')
    onImageReady?.(url)
  }

  const handleFileInput = (e) => handleFile(e.target.files[0])

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleReset = () => {
    stopCamera()
    setPreview(null)
    setMode(null)
    setError(null)
    onImageReady?.(null)
  }

  if (mode === null) {
    return (
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`relative w-full rounded-3xl transition-all duration-200 ${isDragging
          ? 'bg-[#f06292]/10 border-2 border-dashed border-[#f06292]'
          : 'bg-[#d8d8d1] border-2 border-transparent'}`}
        style={{ minHeight: 260 }}
      >
        <canvas ref={canvasRef} className="hidden" />
        <div className="flex flex-col items-center justify-center py-14 px-6 gap-4">
          <label className="cursor-pointer group">
            <div className="w-20 h-20 rounded-full bg-[#e6e6df] flex items-center justify-center shadow-sm group-hover:bg-[#f06292]/10 transition-colors">
              <svg className="w-9 h-9 text-[#f06292]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z" />
              </svg>            </div>
            <input type="file" accept="image/*" onChange={handleFileInput} className="hidden" />
          </label>
          <div className="text-center">
            <p className="text-[#1a1a1a] font-bold text-xl">Drop your dog photo here</p>
            <button
              onClick={() => { setMode('camera'); startCamera() }}
              className="text-[#f06292] text-sm font-medium hover:underline mt-1"
            >
              or Take a Shot to start the scanning
            </button>
          </div>
          <div className="flex items-center gap-2 text-[#888880] text-sm">
            <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            JPEG, PNG, AVIF, WEBP supported
          </div>
        </div>
      </div>
    )
  }

  if (mode === 'camera') {
    return (
      <div className="flex flex-col gap-3 w-full">
        <canvas ref={canvasRef} className="hidden" />
        {error ? (
          <div className="bg-red-50 border border-red-200 text-red-500 rounded-2xl p-4 text-sm text-center">
            ⚠️ {error}
          </div>
        ) : (
          <div className="relative rounded-3xl overflow-hidden bg-black aspect-video">
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-44 h-44 border-4 border-white/70 rounded-full" />
            </div>
            <p className="absolute bottom-3 left-0 right-0 text-center text-white/60 text-xs">
              Arrange the dogs within the circle
            </p>
          </div>
        )}
        <div className="flex gap-2">
          {!error && (
            <button onClick={capturePhoto} className="flex-1 bg-[#f06292] hover:bg-[#e0517e] text-white font-bold py-3 rounded-2xl transition active:scale-95 flex items-center justify-center gap-2">
              shoot
            </button>
          )}
          {hasMultipleCameras && !error && (
            <button onClick={switchCamera} className="bg-[#d8d8d1] hover:bg-[#c8c8c1] text-[#1a1a1a] px-4 rounded-2xl transition active:scale-95">
              Switch
            </button>
          )}
          <button onClick={handleReset} className="bg-[#d8d8d1] hover:bg-[#c8c8c1] text-[#1a1a1a] px-4 rounded-2xl transition active:scale-95">
            ✕
          </button>
        </div>
      </div>
    )
  }

  if (mode === 'preview') {
    return (
      <div className="flex flex-col gap-3 w-full">
        <div className="relative rounded-3xl overflow-hidden aspect-video bg-[#d8d8d1]">
          <img src={preview} alt="preview" className="w-full h-full object-cover" />
        </div>
        <button onClick={handleReset} className="text-sm text-[#888880] hover:text-[#f06292] transition text-center">
          Choose a new image.
        </button>
      </div>
    )
  }
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Homepage() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [image, setImage] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const imgRef = useRef(null)

  const { ready, predict, loadingProgress } = useDogModel()

  const handleIdentify = async () => {
    if (!imgRef.current || !ready) return
    setLoading(true)
    setResult(null)
    try {
      const res = await predict(imgRef.current)
      setResult(res)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#e6e6df]">

      {/* ── Navbar ── */}
      <header className="w-full px-6 py-4 flex items-center justify-between lg:px-12">
        <a href="#" className="-m-1.5 p-1.5">
          <span className="sr-only">CanineScan</span>
          <img
            alt=""
            src="https://res.cloudinary.com/dla8rkqp6/image/upload/v1774107812/w8txyorvtl0bouigh6to.png"
            className="h-10 w-auto"
          />
        </a>


        <button className="lg:hidden p-2 text-[#555]" onClick={() => setMobileMenuOpen(true)}>
          <Bars3Icon className="w-6 h-6" />
        </button>
      </header>

      {/* Mobile Menu */}
      <Dialog open={mobileMenuOpen} onClose={setMobileMenuOpen} className="lg:hidden">
        <div className="fixed inset-0 z-50 bg-black/20" />
        <DialogPanel className="fixed inset-y-0 right-0 z-50 w-full max-w-sm bg-[#e6e6df] p-6 shadow-xl">
          <div className="flex items-center justify-between mb-8">
            <img src="https://res.cloudinary.com/dla8rkqp6/image/upload/v1774107812/w8txyorvtl0bouigh6to.png" className="h-8 w-auto" alt="" />
            <button onClick={() => setMobileMenuOpen(false)} className="text-[#666]">
              <XMarkIcon className="w-6 h-6" />
            </button>
          </div>

        </DialogPanel>
      </Dialog>

      {/* ── Hero ── */}
      <main className="max-w-2xl mx-auto px-6 pt-10 pb-24 flex flex-col items-center text-center gap-8">
        <div>
          <h1 className="text-5xl sm:text-6xl font-extrabold text-[#1a1a1a] leading-tight tracking-tight">
            Identify Any <span className="text-[#f06292]">Pup</span>
            <br />With a Single Snap.
          </h1>
          <p className="mt-5 text-[#666] text-base sm:text-lg leading-relaxed max-w-lg mx-auto">
            Our advanced AI recognizes over 120+ breeds. Simply upload a photo or use your
            camera to discover the unique heritage of your canine companion.
          </p>
        </div>

        {/* Drop Zone */}
        <div className="w-full">
          <DogImageCapture onImageReady={(img) => {
            setImage(img)
            setResult(null)
          }} />        </div>

        {/* Hidden img สำหรับ TF.js */}
        {image && (
          <img ref={imgRef} src={image} alt="dog" className="hidden" crossOrigin="anonymous" />
        )}

        {/* Identify Button */}
        <button
          onClick={handleIdentify}
          disabled={!image || !ready || loading}
          className={`flex items-center gap-3 px-10 py-4 rounded-full font-bold text-base transition-all duration-200 ${image && ready && !loading
            ? 'bg-[#8b1a4a] hover:bg-[#7a1640] text-white shadow-lg hover:shadow-xl active:scale-95 cursor-pointer'
            : 'bg-[#c0a0ad] text-white cursor-not-allowed opacity-70'
            }`}
        >
          {loading ? (
            <>
              <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
              Analyzing...
            </>
          ) : !ready ? (
            <>
              <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
              </svg>
              Loading model {loadingProgress}%
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Identify Breed
            </>
          )}
        </button>
        {/* ผลลัพธ์ */}
        {result && (
          <div className="w-full bg-[#d8d8d1] rounded-3xl p-6 text-left">
            <p className="text-2xl font-bold text-[#1a1a1a] capitalize">{result.breed}</p>
            <p className="text-[#f06292] font-semibold mt-1">
              {(result.confidence * 100).toFixed(1)}% confidence
            </p>
            <div className="mt-4 flex flex-col gap-3">
              {result.top3.map((item) => (
                <div key={item.breed} className="flex items-center gap-3">
                  <span className="text-sm text-[#555] w-40 truncate capitalize">{item.breed}</span>
                  <div className="flex-1 bg-[#bdbdae] rounded-full h-2">
                    <div
                      className="bg-[#f06292] h-2 rounded-full transition-all"
                      style={{ width: `${(item.confidence * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="text-xs text-[#888] w-10 text-right">
                    {(item.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="flex items-center justify-center gap-8 text-center">
          <div>
            <p className="text-2xl font-bold text-[#888880]">120+</p>
            <p className="text-xs text-[#BDBDAE]">Breeds</p>
          </div>
          <div className="w-px h-8 bg-[#BDBDAE]" />
          <div>
            <p className="text-2xl font-bold text-[#888880]">83%</p>
            <p className="text-xs text-[#BDBDAE]">Accuracy</p>
          </div>
          <div className="w-px h-8 bg-[#BDBDAE]" />
          <div>
            <p className="text-2xl font-bold text-[#888880]">AI</p>
            <p className="text-xs text-[#BDBDAE]">Powered</p>
          </div>
        </div>
        {/* 
        <Marquee speed={50} gradient={false} className="w-full text-sm text-[#888]">
          🐶 Golden Retriever &nbsp; 🐶 Labrador &nbsp; 🐶 Poodle &nbsp;
        </Marquee> */}

      </main>
    </div>
  )
}