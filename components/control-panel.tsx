"use client"

import type React from "react"

import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Upload, Play, Square, AlertCircle, CheckCircle } from "lucide-react"

interface ControlPanelProps {
  onUploadOHLCV: (file: File) => Promise<void>
  onUploadNotebook: (file: File) => Promise<void>
  onRunBacktest: () => void
  isRunning: boolean
  isConnected: boolean
}

export function ControlPanel({
  onUploadOHLCV,
  onUploadNotebook,
  onRunBacktest,
  isRunning,
  isConnected,
}: ControlPanelProps) {
  const ohlcvInputRef = useRef<HTMLInputElement>(null)
  const notebookInputRef = useRef<HTMLInputElement>(null)
  const [uploadStatus, setUploadStatus] = useState<{ type: "success" | "error"; message: string } | null>(null)

  const handleOHLCVUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      try {
        await onUploadOHLCV(file)
        setUploadStatus({ type: "success", message: "OHLCV data uploaded successfully" })
        setTimeout(() => setUploadStatus(null), 3000)
      } catch (error) {
        setUploadStatus({ type: "error", message: "Failed to upload OHLCV data" })
      }
    }
  }

  const handleNotebookUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      try {
        await onUploadNotebook(file)
        setUploadStatus({ type: "success", message: "Notebook uploaded successfully" })
        setTimeout(() => setUploadStatus(null), 3000)
      } catch (error) {
        setUploadStatus({ type: "error", message: "Failed to upload notebook" })
      }
    }
  }

  return (
    <div className="p-4 space-y-4">
      <div className="space-y-2">
        <h2 className="font-semibold text-sm">Data Upload</h2>

        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start bg-transparent"
          onClick={() => ohlcvInputRef.current?.click()}
        >
          <Upload className="w-4 h-4 mr-2" />
          Upload OHLCV
        </Button>
        <input ref={ohlcvInputRef} type="file" accept=".csv,.json" onChange={handleOHLCVUpload} className="hidden" />

        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start bg-transparent"
          onClick={() => notebookInputRef.current?.click()}
        >
          <Upload className="w-4 h-4 mr-2" />
          Upload Notebook
        </Button>
        <input ref={notebookInputRef} type="file" accept=".ipynb" onChange={handleNotebookUpload} className="hidden" />
      </div>

      {uploadStatus && (
        <div
          className={`p-2 rounded text-xs flex items-center gap-2 ${
            uploadStatus.type === "success" ? "bg-green-900 text-green-200" : "bg-red-900 text-red-200"
          }`}
        >
          {uploadStatus.type === "success" ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
          {uploadStatus.message}
        </div>
      )}

      <div className="space-y-2">
        <h2 className="font-semibold text-sm">Backtest Control</h2>

        <Button
          size="sm"
          className="w-full justify-start"
          onClick={onRunBacktest}
          variant={isRunning ? "destructive" : "default"}
        >
          {isRunning ? (
            <>
              <Square className="w-4 h-4 mr-2" />
              Stop Backtest
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Run Backtest
            </>
          )}
        </Button>

        <div className="text-xs space-y-1 p-2 bg-slate-800 rounded">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`} />
            <span>{isConnected ? "Connected" : "Disconnected"}</span>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <h2 className="font-semibold text-sm">Info</h2>
        <div className="text-xs text-muted-foreground space-y-1">
          <p>1. Upload OHLCV data (CSV/JSON)</p>
          <p>2. Upload trading strategy notebook</p>
          <p>3. Click Run Backtest</p>
          <p>4. Watch real-time visualization</p>
        </div>
      </div>
    </div>
  )
}
