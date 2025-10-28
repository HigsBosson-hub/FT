"use client"

import { useEffect, useRef } from "react"
import { createChart, ColorType } from "lightweight-charts"
import type { ChartData } from "@/types/chart"

interface ChartContainerProps {
  data: ChartData
}

export function ChartContainer({ data }: ChartContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const candleSeriesRef = useRef<any>(null)
  const lineSeriesRef = useRef<Map<string, any>>(new Map())

  useEffect(() => {
    if (!containerRef.current) return

    // Create chart
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f172a" },
        textColor: "#cbd5e1",
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    })

    chartRef.current = chart

    // Create candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#ef4444",
      borderUpColor: "#10b981",
      borderDownColor: "#ef4444",
      wickUpColor: "#10b981",
      wickDownColor: "#ef4444",
    })

    candleSeriesRef.current = candleSeries

    // Add candlesticks
    if (data.candlesticks.length > 0) {
      candleSeries.setData(data.candlesticks)
      chart.timeScale().fitContent()
    }

    data.indicators.forEach((indicator) => {
      if (!lineSeriesRef.current.has(indicator.name)) {
        const lineSeries = chart.addLineSeries({
          color: indicator.color || "#3b82f6",
          lineWidth: 2,
          title: indicator.name,
        })
        lineSeriesRef.current.set(indicator.name, lineSeries)
        lineSeries.setData(indicator.values)
      }
    })

    data.boxes.forEach((box) => {
      const boxSeries = chart.addAreaSeries({
        lineColor: box.color || "#8b5cf6",
        topColor: (box.color || "#8b5cf6") + "33",
        bottomColor: (box.color || "#8b5cf6") + "0a",
        lineWidth: 1,
      })
      boxSeries.setData([
        { time: box.startTime, value: box.topPrice },
        { time: box.endTime, value: box.topPrice },
      ])
    })

    // Add signals as markers
    if (data.signals.length > 0) {
      const markers = data.signals.map((signal) => ({
        time: signal.time,
        position: signal.type === "BUY" ? ("belowBar" as const) : ("aboveBar" as const),
        color: signal.type === "BUY" ? "#10b981" : "#ef4444",
        shape: signal.type === "BUY" ? ("arrowUp" as const) : ("arrowDown" as const),
        text: signal.type,
      }))
      candleSeries.setMarkers(markers)
    }

    // Handle resize
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      chart.remove()
    }
  }, [data.candlesticks, data.signals, data.indicators, data.boxes])

  return <div ref={containerRef} className="flex-1 bg-slate-900" style={{ minHeight: "400px" }} />
}
