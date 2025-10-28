"use client"

import { useEffect, useRef, useState } from "react"
import { Copy, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface LogConsoleProps {
  logs: string[]
}

export function LogConsole({ logs }: LogConsoleProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleCopyLogs = () => {
    const logsText = logs.join("\n")
    navigator.clipboard.writeText(logsText)
  }

  const handleClearLogs = () => {
    // This would need to be passed as a prop from parent
    console.log("Clear logs")
  }

  const getLogColor = (log: string) => {
    if (log.includes("ERROR") || log.includes("error")) return "text-red-400"
    if (log.includes("WARNING") || log.includes("warning")) return "text-yellow-400"
    if (log.includes("SUCCESS") || log.includes("success")) return "text-green-400"
    if (log.includes("INFO") || log.includes("info")) return "text-blue-400"
    return "text-slate-400"
  }

  return (
    <div className="flex flex-col h-full bg-slate-900">
      <div className="p-3 border-b border-border flex items-center justify-between">
        <h2 className="font-semibold text-sm">Live Log</h2>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={handleCopyLogs} title="Copy logs">
            <Copy className="w-3 h-3" />
          </Button>
          <Button size="sm" variant="ghost" className="h-6 w-6 p-0" onClick={handleClearLogs} title="Clear logs">
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      </div>

      <div className="px-3 py-2 border-b border-border flex items-center gap-2 text-xs">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="w-3 h-3"
          />
          Auto-scroll
        </label>
        <span className="text-muted-foreground">({logs.length} lines)</span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-1 font-mono text-xs">
        {logs.length === 0 ? (
          <div className="text-muted-foreground">Waiting for backtest to start...</div>
        ) : (
          logs.map((log, idx) => (
            <div key={idx} className={getLogColor(log)}>
              {log}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
