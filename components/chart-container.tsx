"use client"

import { useEffect, useRef } from "react"
import { createChart, ColorType, LineStyle } from "lightweight-charts"
import type { ChartData } from "@/types/chart"

interface ChartContainerProps {
  data: ChartData
}

export function ChartContainer({ data }: ChartContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<any>(null)
  const candleSeriesRef = useRef<any>(null)
  const lineSeriesRef = useRef<Map<string, any>>(new Map())
  const atrSeriesRef = useRef<any>(null)
  const volSeriesRef = useRef<any>(null)
  const channelSeriesRef = useRef<any[]>([])

  useEffect(() => {
    if (!containerRef.current) return

    // Create main chart
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0a0e27" },
        textColor: "#d1d4dc",
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: "#2B2B43",
      },
      grid: {
        vertLines: { color: "#1a1e3a", style: LineStyle.Solid },
        horzLines: { color: "#1a1e3a", style: LineStyle.Solid },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          width: 1,
          color: "#758696",
          style: LineStyle.Dashed,
        },
        horzLine: {
          width: 1,
          color: "#758696",
          style: LineStyle.Dashed,
        },
      },
    })

    chartRef.current = chart

    // Create candlestick series with custom colors
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderUpColor: "#26a69a",
      borderDownColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
    })

    candleSeriesRef.current = candleSeries

    // Add candlestick data
    if (data.candlesticks.length > 0) {
      candleSeries.setData(data.candlesticks)
      chart.timeScale().fitContent()
    }

    // Add ATR indicator (cyan line)
    const atrIndicator = data.indicators.find(ind => ind.name.toLowerCase().includes('atr'))
    if (atrIndicator) {
      const atrSeries = chart.addLineSeries({
        color: "#00bcd4",
        lineWidth: 2,
        title: "ATR",
        priceLineVisible: false,
        lastValueVisible: true,
      })
      atrSeries.setData(atrIndicator.values)
      atrSeriesRef.current = atrSeries
    }

    // Add Volatility (VOL) indicator (orange line)
    const volIndicator = data.indicators.find(ind => ind.name.toLowerCase().includes('vol') && !ind.name.toLowerCase().includes('volume'))
    if (volIndicator) {
      const volSeries = chart.addLineSeries({
        color: "#ff9800",
        lineWidth: 2,
        title: "Volatility",
        priceLineVisible: false,
        lastValueVisible: true,
      })
      volSeries.setData(volIndicator.values)
      volSeriesRef.current = volSeries
    }

    // Add other indicators (Upper/Lower bounds, etc.)
    data.indicators.forEach((indicator) => {
      if (indicator.name.toLowerCase().includes('atr') || 
          (indicator.name.toLowerCase().includes('vol') && !indicator.name.toLowerCase().includes('volume'))) {
        return // Already handled above
      }

      if (!lineSeriesRef.current.has(indicator.name)) {
        let color = indicator.color || "#3b82f6"
        let lineStyle = LineStyle.Solid
        
        // Color coding for upper/lower bounds
        if (indicator.name.toLowerCase().includes('upper')) {
          color = "#9c27b0" // Purple for upper
          lineStyle = LineStyle.Dashed
        } else if (indicator.name.toLowerCase().includes('lower')) {
          color = "#ff5722" // Deep orange for lower
          lineStyle = LineStyle.Dashed
        } else if (indicator.name.toLowerCase().includes('channel high')) {
          color = "#4caf50" // Green for channel high
        } else if (indicator.name.toLowerCase().includes('channel low')) {
          color = "#f44336" // Red for channel low
        }

        const lineSeries = chart.addLineSeries({
          color: color,
          lineWidth: 2,
          lineStyle: lineStyle,
          title: indicator.name,
          priceLineVisible: false,
          lastValueVisible: true,
        })
        lineSeriesRef.current.set(indicator.name, lineSeries)
        lineSeries.setData(indicator.values)
      }
    })

    // Draw breakout channels (boxes)
    data.boxes.forEach((box) => {
      // Create upper boundary
      const upperSeries = chart.addLineSeries({
        color: box.color || "#4caf50",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
      })
      upperSeries.setData([
        { time: box.startTime, value: box.topPrice },
        { time: box.endTime, value: box.topPrice },
      ])
      channelSeriesRef.current.push(upperSeries)

      // Create lower boundary
      const lowerSeries = chart.addLineSeries({
        color: box.color || "#f44336",
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        priceLineVisible: false,
      })
      lowerSeries.setData([
        { time: box.startTime, value: box.bottomPrice },
        { time: box.endTime, value: box.bottomPrice },
      ])
      channelSeriesRef.current.push(lowerSeries)

      // Create shaded area
      const areaSeries = chart.addAreaSeries({
        lineColor: "transparent",
        topColor: (box.color || "#8b5cf6") + "22",
        bottomColor: (box.color || "#8b5cf6") + "05",
        lineWidth: 0,
        priceLineVisible: false,
      })
      areaSeries.setData([
        { time: box.startTime, value: box.topPrice },
        { time: box.endTime, value: box.topPrice },
      ])
      channelSeriesRef.current.push(areaSeries)
    })

    // Add trade signals as markers with TP/SL levels
    if (data.signals.length > 0) {
      const markers = data.signals.map((signal) => {
        const isBuy = signal.type === "BUY"
        return {
          time: signal.time,
          position: isBuy ? ("belowBar" as const) : ("aboveBar" as const),
          color: isBuy ? "#26a69a" : "#ef5350",
          shape: isBuy ? ("arrowUp" as const) : ("arrowDown" as const),
          text: `${signal.type} @ ${signal.price.toFixed(2)}`,
          size: 2,
        }
      })
      candleSeries.setMarkers(markers)

      // Add TP/SL levels as horizontal lines if available in signal data
      data.signals.forEach((signal: any) => {
        if (signal.tp1) {
          const tp1Series = chart.addLineSeries({
            color: "#4caf50",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
          })
          tp1Series.setData([
            { time: signal.time, value: signal.tp1 },
            { time: signal.time + 3600, value: signal.tp1 }, // Extend 1 hour
          ])
          channelSeriesRef.current.push(tp1Series)
        }
        
        if (signal.tp2) {
          const tp2Series = chart.addLineSeries({
            color: "#8bc34a",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
          })
          tp2Series.setData([
            { time: signal.time, value: signal.tp2 },
            { time: signal.time + 3600, value: signal.tp2 },
          ])
          channelSeriesRef.current.push(tp2Series)
        }
        
        if (signal.sl) {
          const slSeries = chart.addLineSeries({
            color: "#f44336",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
          })
          slSeries.setData([
            { time: signal.time, value: signal.sl },
            { time: signal.time + 3600, value: signal.sl },
          ])
          channelSeriesRef.current.push(slSeries)
        }
      })
    }

    // Add legend
    const legend = document.createElement('div')
    legend.style.cssText = `
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 10;
      background: rgba(10, 14, 39, 0.85);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-family: monospace;
      color: #d1d4dc;
      border: 1px solid #2B2B43;
    `
    legend.innerHTML = `
      <div style="margin-bottom: 4px; font-weight: bold; color: #00bcd4;">BOCS Adaptive Strategy</div>
      <div style="display: flex; gap: 16px; flex-wrap: wrap;">
        <div><span style="color: #00bcd4;">●</span> ATR</div>
        <div><span style="color: #ff9800;">●</span> Volatility</div>
        <div><span style="color: #9c27b0;">- -</span> Upper</div>
        <div><span style="color: #ff5722;">- -</span> Lower</div>
        <div><span style="color: #26a69a;">▲</span> Long</div>
        <div><span style="color: #ef5350;">▼</span> Short</div>
      </div>
    `
    containerRef.current.appendChild(legend)

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

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 relative">
        <div ref={containerRef} className="w-full h-full" style={{ minHeight: "500px" }} />
      </div>
    </div>
  )
}