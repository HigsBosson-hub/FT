// types/websocket.ts
export interface WebSocketMessage {
  type: 'candlestick' | 'signal' | 'indicator' | 'box' | 'metrics' | 'log' | 'channel' | 'volatility';
  payload: any;
}

export interface ChannelData {
  startTime: number;
  endTime: number;
  upperBound: number;
  lowerBound: number;
  volatilityAdjustment: number;
  type: 'upper_cross_lower' | 'lower_cross_upper';
}

export interface VolatilityData {
  time: number;
  atr: number;
  vol: number;
  normalizedPrice: number;
  upper: number;
  lower: number;
}

export interface TradeMetrics {
  equity: number;
  pnl: number;
  pnlDollars: number;
  drawdown: number;
  winRate: number;
  trades: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
  totalWins?: number;
  totalLosses?: number;
}