"use client"

import { TrendingUp, TrendingDown } from "lucide-react"

interface MetricsPanelProps {
  metrics: {
    equity: number
    pnl: number
    drawdown: number
    winRate: number
    trades: number
  }
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
    }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`
  }

  const getPnlColor = (value: number) => {
    if (value > 0) return "text-green-400"
    if (value < 0) return "text-red-400"
    return "text-slate-400"
  }

  const getDrawdownColor = (value: number) => {
    if (value < 0.1) return "text-green-400"
    if (value < 0.2) return "text-yellow-400"
    return "text-red-400"
  }

  const getWinRateColor = (value: number) => {
    if (value > 0.6) return "text-green-400"
    if (value > 0.4) return "text-yellow-400"
    return "text-red-400"
  }

  return (
    <div className="p-4 space-y-4">
      <h2 className="font-semibold text-sm">Performance Metrics</h2>

      <div className="grid grid-cols-2 gap-3">
        {/* Equity */}
        <div className="bg-slate-800 p-3 rounded border border-slate-700 hover:border-slate-600 transition-colors">
          <div className="text-xs text-muted-foreground">Equity</div>
          <div className="text-lg font-semibold text-green-400">{formatCurrency(metrics.equity)}</div>
        </div>

        {/* P&L */}
        <div className="bg-slate-800 p-3 rounded border border-slate-700 hover:border-slate-600 transition-colors">
          <div className="flex items-center justify-between">
            <div className="text-xs text-muted-foreground">P&L</div>
            {metrics.pnl >= 0 ? (
              <TrendingUp className="w-3 h-3 text-green-400" />
            ) : (
              <TrendingDown className="w-3 h-3 text-red-400" />
            )}
          </div>
          <div className={`text-lg font-semibold ${getPnlColor(metrics.pnl)}`}>{formatCurrency(metrics.pnl)}</div>
        </div>

        {/* Drawdown */}
        <div className="bg-slate-800 p-3 rounded border border-slate-700 hover:border-slate-600 transition-colors">
          <div className="text-xs text-muted-foreground">Drawdown</div>
          <div className={`text-lg font-semibold ${getDrawdownColor(metrics.drawdown)}`}>
            {formatPercent(metrics.drawdown)}
          </div>
        </div>

        {/* Win Rate */}
        <div className="bg-slate-800 p-3 rounded border border-slate-700 hover:border-slate-600 transition-colors">
          <div className="text-xs text-muted-foreground">Win Rate</div>
          <div className={`text-lg font-semibold ${getWinRateColor(metrics.winRate)}`}>
            {formatPercent(metrics.winRate)}
          </div>
        </div>

        {/* Total Trades */}
        <div className="col-span-2 bg-slate-800 p-3 rounded border border-slate-700 hover:border-slate-600 transition-colors">
          <div className="text-xs text-muted-foreground">Total Trades</div>
          <div className="text-lg font-semibold">{metrics.trades}</div>
        </div>
      </div>
    </div>
  )
}
