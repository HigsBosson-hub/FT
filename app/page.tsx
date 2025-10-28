"use client"

import { useState, useCallback } from "react"
import { ChartContainer } from "@/components/chart-container"
import { ControlPanel } from "@/components/control-panel"
import { MetricsPanel } from "@/components/metrics-panel"
import { LogConsole } from "@/components/log-console"
import { useWebSocket } from "@/hooks/use-websocket"
import { useChart } from "@/hooks/use-chart"
import { useFileUpload } from "@/hooks/use-file-upload"

export default function Home() {
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [metrics, setMetrics] = useState({
    equity: 100000,
    pnl: 0,
    drawdown: 0,
    winRate: 0,
    trades: 0,
  })

  const { chartData, addCandlestick, addSignal, addIndicator, addBox, clearChart } = useChart()
  const { uploadOHLCV, uploadNotebook } = useFileUpload()

  const { connect, disconnect, isConnected } = useWebSocket({
    onMessage: (data) => {
      try {
        if (data.type === "candlestick") {
          addCandlestick(data.payload)
        } else if (data.type === "signal") {
          addSignal(data.payload)
        } else if (data.type === "indicator") {
          addIndicator(data.payload)
        } else if (data.type === "box") {
          addBox(data.payload)
        } else if (data.type === "metrics") {
          setMetrics(data.payload)
        } else if (data.type === "log") {
          setLogs((prev) => [...prev, data.payload])
        }
      } catch (error) {
        console.error("Error processing WebSocket message:", error)
        setLogs((prev) => [...prev, `ERROR: Failed to process message - ${error}`])
      }
    },
    onConnect: () => {
      setLogs((prev) => [...prev, "INFO: WebSocket connected"])
    },
    onDisconnect: () => {
      setLogs((prev) => [...prev, "WARNING: WebSocket disconnected"])
    },
    onError: (error) => {
      setLogs((prev) => [...prev, `ERROR: ${error.message}`])
    },
  })

  const handleRunBacktest = useCallback(async () => {
    if (!isRunning) {
      setIsRunning(true)
      clearChart()
      setLogs([])
      setMetrics({
        equity: 100000,
        pnl: 0,
        drawdown: 0,
        winRate: 0,
        trades: 0,
      })
      connect("ws://localhost:8000/ws")
      setLogs((prev) => [...prev, "INFO: Starting backtest..."])
    } else {
      setIsRunning(false)
      disconnect()
      setLogs((prev) => [...prev, "INFO: Backtest stopped"])
    }
  }, [isRunning, clearChart, connect, disconnect])

  const handleUploadOHLCV = useCallback(
    async (file: File) => {
      try {
        setLogs((prev) => [...prev, `INFO: Uploading OHLCV file: ${file.name}`])
        await uploadOHLCV(file)
        setLogs((prev) => [...prev, `SUCCESS: OHLCV file uploaded successfully`])
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : "Unknown error"
        setLogs((prev) => [...prev, `ERROR: Failed to upload OHLCV - ${errorMsg}`])
      }
    },
    [uploadOHLCV],
  )

  const handleUploadNotebook = useCallback(
    async (file: File) => {
      try {
        setLogs((prev) => [...prev, `INFO: Uploading notebook file: ${file.name}`])
        await uploadNotebook(file)
        setLogs((prev) => [...prev, `SUCCESS: Notebook file uploaded successfully`])
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : "Unknown error"
        setLogs((prev) => [...prev, `ERROR: Failed to upload notebook - ${errorMsg}`])
      }
    },
    [uploadNotebook],
  )

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="flex flex-col h-screen">
        {/* Header */}
        <header className="border-b border-border bg-card p-4 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Trading Backtest Visualizer</h1>
              <p className="text-sm text-muted-foreground">Real-time OHLCV visualization with trading signals</p>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={`flex items-center gap-2 px-3 py-1 rounded text-sm ${isConnected ? "bg-green-900 text-green-200" : "bg-red-900 text-red-200"}`}
              >
                <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-400" : "bg-red-400"}`} />
                {isConnected ? "Connected" : "Disconnected"}
              </div>
              <div
                className={`flex items-center gap-2 px-3 py-1 rounded text-sm ${isRunning ? "bg-blue-900 text-blue-200" : "bg-slate-700 text-slate-300"}`}
              >
                <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-blue-400 animate-pulse" : "bg-slate-400"}`} />
                {isRunning ? "Running" : "Idle"}
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Sidebar - Controls */}
          <div className="w-64 border-r border-border bg-card overflow-y-auto">
            <ControlPanel
              onUploadOHLCV={handleUploadOHLCV}
              onUploadNotebook={handleUploadNotebook}
              onRunBacktest={handleRunBacktest}
              isRunning={isRunning}
              isConnected={isConnected}
            />
          </div>

          {/* Center - Chart */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <ChartContainer data={chartData} />

            {/* Bottom - Metrics & Logs */}
            <div className="flex border-t border-border bg-card">
              <div className="flex-1 border-r border-border overflow-y-auto">
                <MetricsPanel metrics={metrics} />
              </div>
              <div className="w-80">
                <LogConsole logs={logs} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
