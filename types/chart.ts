export interface Candlestick {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface Signal {
  time: number
  type: "BUY" | "SELL"
  price: number
}

export interface Indicator {
  name: string
  values: Array<{
    time: number
    value: number
  }>
  color?: string
}

export interface Box {
  startTime: number
  endTime: number
  topPrice: number
  bottomPrice: number
  color?: string
}

export interface ChartData {
  candlesticks: Candlestick[]
  signals: Signal[]
  indicators: Indicator[]
  boxes: Box[]
}
