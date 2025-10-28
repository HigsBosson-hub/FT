"use client"

import { useState, useCallback } from "react"
import type { ChartData, Candlestick, Signal, Indicator, Box } from "@/types/chart"

export function useChart() {
  const [chartData, setChartData] = useState<ChartData>({
    candlesticks: [],
    signals: [],
    indicators: [],
    boxes: [],
  })

  const addCandlestick = useCallback((candlestick: Candlestick) => {
    setChartData((prev) => ({
      ...prev,
      candlesticks: [...prev.candlesticks, candlestick],
    }))
  }, [])

  const addCandlesticks = useCallback((candlesticks: Candlestick[]) => {
    setChartData((prev) => ({
      ...prev,
      candlesticks: [...prev.candlesticks, ...candlesticks],
    }))
  }, [])

  const addSignal = useCallback((signal: Signal) => {
    setChartData((prev) => ({
      ...prev,
      signals: [...prev.signals, signal],
    }))
  }, [])

  const addSignals = useCallback((signals: Signal[]) => {
    setChartData((prev) => ({
      ...prev,
      signals: [...prev.signals, ...signals],
    }))
  }, [])

  const addIndicator = useCallback((indicator: Indicator) => {
    setChartData((prev) => {
      const existingIndex = prev.indicators.findIndex((ind) => ind.name === indicator.name)
      if (existingIndex >= 0) {
        const updated = [...prev.indicators]
        updated[existingIndex] = indicator
        return { ...prev, indicators: updated }
      }
      return {
        ...prev,
        indicators: [...prev.indicators, indicator],
      }
    })
  }, [])

  const addIndicators = useCallback((indicators: Indicator[]) => {
    setChartData((prev) => {
      const updated = [...prev.indicators]
      indicators.forEach((indicator) => {
        const existingIndex = updated.findIndex((ind) => ind.name === indicator.name)
        if (existingIndex >= 0) {
          updated[existingIndex] = indicator
        } else {
          updated.push(indicator)
        }
      })
      return { ...prev, indicators: updated }
    })
  }, [])

  const addBox = useCallback((box: Box) => {
    setChartData((prev) => ({
      ...prev,
      boxes: [...prev.boxes, box],
    }))
  }, [])

  const addBoxes = useCallback((boxes: Box[]) => {
    setChartData((prev) => ({
      ...prev,
      boxes: [...prev.boxes, ...boxes],
    }))
  }, [])

  const clearChart = useCallback(() => {
    setChartData({
      candlesticks: [],
      signals: [],
      indicators: [],
      boxes: [],
    })
  }, [])

  const clearCandlesticks = useCallback(() => {
    setChartData((prev) => ({
      ...prev,
      candlesticks: [],
    }))
  }, [])

  const clearSignals = useCallback(() => {
    setChartData((prev) => ({
      ...prev,
      signals: [],
    }))
  }, [])

  const clearIndicators = useCallback(() => {
    setChartData((prev) => ({
      ...prev,
      indicators: [],
    }))
  }, [])

  const clearBoxes = useCallback(() => {
    setChartData((prev) => ({
      ...prev,
      boxes: [],
    }))
  }, [])

  const removeIndicator = useCallback((name: string) => {
    setChartData((prev) => ({
      ...prev,
      indicators: prev.indicators.filter((ind) => ind.name !== name),
    }))
  }, [])

  return {
    chartData,
    addCandlestick,
    addCandlesticks,
    addSignal,
    addSignals,
    addIndicator,
    addIndicators,
    addBox,
    addBoxes,
    clearChart,
    clearCandlesticks,
    clearSignals,
    clearIndicators,
    clearBoxes,
    removeIndicator,
  }
}
