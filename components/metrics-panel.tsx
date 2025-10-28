"use client"

import { TrendingUp, TrendingDown, Activity, Target, AlertTriangle, BarChart3 } from "lucide-react"

interface MetricsPanelProps {
  metrics: {
    equity: number
    pnl: number
    drawdown: number
    winRate: number
    trades: number
    sharpeRatio?: number
    maxDrawdown?: number
    totalWins?: number
    totalLosses?: number
    avgWin?: number
    avgLoss?: number
    profitFactor?: number
    currentATR?: number
    currentVol?: number
  }
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`
  }

  const formatNumber = (value: number, decimals: number = 2) => {
    return value.toFixed(decimals)
  }

  const getPnlColor = (value: number) => {
    if (value > 0) return "text-emerald-400"
    if (value < 0) return "text-red-400"
    return "text-slate-400"
  }

  const getDrawdownColor = (value: number) => {
    if (value < 0.1) return "text-emerald-400"
    if (value < 0.2) return "text-amber-400"
    return "text-red-400"
  }

  const getWinRateColor = (value: number) => {
    if (value > 0.6) return "text-emerald-400"
    if (value > 0.4) return "text-amber-400"
    return "text-red-400"
  }

  const getSharpeColor = (value: number) => {
    if (value > 2) return "text-emerald-400"
    if (value > 1) return "text-amber-400"
    return "text-slate-400"
  }

  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-sm flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          BOCS Strategy Metrics
        </h2>
      </div>

      {/* Primary Metrics */}
      <div className="grid grid-cols-2 gap-3">
        {/* Equity */}
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800 hover:border-cyan-500/30 transition-all">
          <div className="text-xs text-slate-400 mb-1">Portfolio Equity</div>
          <div className="text-lg font-bold text-cyan-400">{formatCurrency(metrics.equity)}</div>
        </div>

        {/* P&L */}
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800 hover:border-cyan-500/30 transition-all">
          <div className="flex items-center justify-between mb-1">
            <div className="text-xs text-slate-400">Net P&L</div>
            {metrics.pnl >= 0 ? (
              <TrendingUp className="w-3 h-3 text-emerald-400" />
            ) : (
              <TrendingDown className="w-3 h-3 text-red-400" />
            )}
          </div>
          <div className={`text-lg font-bold ${getPnlColor(metrics.pnl)}`}>
            {formatCurrency(metrics.pnl)}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {metrics.pnl >= 0 ? '+' : ''}{formatPercent(metrics.pnl / 100000)}
          </div>
        </div>

        {/* Total Trades */}
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800 hover:border-purple-500/30 transition-all">
          <div className="text-xs text-slate-400 mb-1">Total Trades</div>
          <div className="text-lg font-bold text-purple-400">{metrics.trades}</div>
          {metrics.totalWins !== undefined && metrics.totalLosses !== undefined && (
            <div className="text-xs text-slate-500 mt-1">
              <span className="text-emerald-400">{metrics.totalWins}W</span> / 
              <span className="text-red-400"> {metrics.totalLosses}L</span>
            </div>
          )}
        </div>

        {/* Win Rate */}
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800 hover:border-purple-500/30 transition-all">
          <div className="text-xs text-slate-400 mb-1">Win Rate</div>
          <div className={`text-lg font-bold ${getWinRateColor(metrics.winRate)}`}>
            {formatPercent(metrics.winRate)}
          </div>
          {metrics.avgWin !== undefined && metrics.avgLoss !== undefined && (
            <div className="text-xs text-slate-500 mt-1">
              Avg: <span className="text-emerald-400">${metrics.avgWin.toFixed(0)}</span> / 
              <span className="text-red-400"> ${Math.abs(metrics.avgLoss).toFixed(0)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-slate-400 flex items-center gap-2">
          <AlertTriangle className="w-3 h-3" />
          Risk Metrics
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {/* Current Drawdown */}
          <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
            <div className="text-xs text-slate-400 mb-1">Current DD</div>
            <div className={`text-base font-bold ${getDrawdownColor(metrics.drawdown)}`}>
              {formatPercent(metrics.drawdown)}
            </div>
          </div>

          {/* Max Drawdown */}
          {metrics.maxDrawdown !== undefined && (
            <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">Max DD</div>
              <div className={`text-base font-bold ${getDrawdownColor(metrics.maxDrawdown)}`}>
                {formatPercent(metrics.maxDrawdown)}
              </div>
            </div>
          )}

          {/* Sharpe Ratio */}
          {metrics.sharpeRatio !== undefined && (
            <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">Sharpe Ratio</div>
              <div className={`text-base font-bold ${getSharpeColor(metrics.sharpeRatio)}`}>
                {formatNumber(metrics.sharpeRatio, 2)}
              </div>
            </div>
          )}

          {/* Profit Factor */}
          {metrics.profitFactor !== undefined && (
            <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">Profit Factor</div>
              <div className={`text-base font-bold ${metrics.profitFactor > 1.5 ? 'text-emerald-400' : 'text-slate-400'}`}>
                {formatNumber(metrics.profitFactor, 2)}x
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Volatility Indicators */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-slate-400 flex items-center gap-2">
          <Activity className="w-3 h-3" />
          Current Volatility
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {/* Current ATR */}
          {metrics.currentATR !== undefined && (
            <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">ATR</div>
              <div className="text-base font-bold text-cyan-400">
                {formatNumber(metrics.currentATR, 2)}
              </div>
            </div>
          )}

          {/* Current VOL */}
          {metrics.currentVol !== undefined && (
            <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">VOL</div>
              <div className="text-base font-bold text-orange-400">
                {formatNumber(metrics.currentVol, 4)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="bg-gradient-to-r from-slate-900/50 to-cyan-900/20 p-3 rounded-lg border border-cyan-800/30">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-slate-400">System Status</div>
            <div className="text-sm font-bold text-cyan-400 mt-1">
              {metrics.trades > 0 ? 'Active Trading' : 'Awaiting Signal'}
            </div>
          </div>
          <Target className="w-6 h-6 text-cyan-400 opacity-50" />
        </div>
      </div>
    </div>
  )
}