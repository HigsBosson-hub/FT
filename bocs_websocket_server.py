"""
BOCS Strategy WebSocket Server
Integrates your existing BOCS strategy with the frontend visualizer
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Import your existing BOCS strategy class
from bocs_strategy import BOCSStrategy


class BOCSWebSocketServer:
    def __init__(self):
        self.clients = set()
        self.strategy = None
        
    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"✓ Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"✗ Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_to_all(self, message):
        """Send message to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def send_message(self, msg_type, payload):
        """Send formatted WebSocket message"""
        message = json.dumps({
            "type": msg_type,
            "payload": payload
        })
        await self.send_to_all(message)
    
    async def send_log(self, message, level="INFO"):
        """Send log message to frontend"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] {level}: {message}"
        await self.send_message("log", log_msg)
    
    async def send_candlestick(self, row):
        """Send candlestick data"""
        await self.send_message("candlestick", {
            "time": int(row['DateTime'].timestamp()),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": float(row.get('Volume', 0))
        })
    
    async def send_indicator(self, name, time, value, color=None):
        """Send indicator update"""
        payload = {
            "name": name,
            "values": [{"time": int(time.timestamp()), "value": float(value)}]
        }
        if color:
            payload["color"] = color
        await self.send_message("indicator", payload)
    
    async def send_channel_box(self, start_time, end_time, high, low, color="#8b5cf6"):
        """Send channel visualization"""
        await self.send_message("box", {
            "startTime": int(start_time.timestamp()),
            "endTime": int(end_time.timestamp()),
            "topPrice": float(high),
            "bottomPrice": float(low),
            "color": color
        })
    
    async def send_signal(self, row, signal_type, entry_price, tp1, tp2, sl):
        """Send trade signal"""
        payload = {
            "time": int(row['DateTime'].timestamp()),
            "type": signal_type,
            "price": float(entry_price)
        }
        if tp1 and not np.isnan(tp1):
            payload["tp1"] = float(tp1)
        if tp2 and not np.isnan(tp2):
            payload["tp2"] = float(tp2)
        if sl and not np.isnan(sl):
            payload["sl"] = float(sl)
        
        await self.send_message("signal", payload)
    
    async def send_metrics(self, equity, trades_list, initial_capital, current_atr, current_vol):
        """Send performance metrics"""
        if len(trades_list) == 0:
            await self.send_message("metrics", {
                "equity": float(initial_capital),
                "pnl": 0.0,
                "drawdown": 0.0,
                "winRate": 0.0,
                "trades": 0,
                "sharpeRatio": 0.0,
                "maxDrawdown": 0.0,
                "totalWins": 0,
                "totalLosses": 0,
                "avgWin": 0.0,
                "avgLoss": 0.0,
                "profitFactor": 0.0,
                "currentATR": float(current_atr) if not np.isnan(current_atr) else 0.0,
                "currentVol": float(current_vol) if not np.isnan(current_vol) else 0.0
            })
            return
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades_list)
        point_value = 1000
        trades_df['pnl_dollars'] = trades_df['pnl'] * point_value
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl_dollars'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_dollars'].mean() if len(losing_trades) > 0 else 0
        
        total_pnl = trades_df['pnl_dollars'].sum()
        
        # Calculate drawdown
        trades_df['cumulative_pnl'] = trades_df['pnl_dollars'].cumsum()
        trades_df['equity_curve'] = initial_capital + trades_df['cumulative_pnl']
        trades_df['peak'] = trades_df['equity_curve'].cummax()
        trades_df['drawdown'] = trades_df['equity_curve'] - trades_df['peak']
        trades_df['drawdown_pct'] = (trades_df['drawdown'] / trades_df['peak'])
        
        max_dd = abs(trades_df['drawdown'].min())
        max_dd_pct = abs(trades_df['drawdown_pct'].min())
        current_dd = abs(trades_df['drawdown'].iloc[-1]) / trades_df['peak'].iloc[-1] if len(trades_df) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl_dollars'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_dollars'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = trades_df['pnl_dollars'].values
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        await self.send_message("metrics", {
            "equity": float(equity),
            "pnl": float(total_pnl),
            "drawdown": float(current_dd),
            "winRate": float(win_rate),
            "trades": int(total_trades),
            "sharpeRatio": float(sharpe),
            "maxDrawdown": float(max_dd_pct),
            "totalWins": int(len(winning_trades)),
            "totalLosses": int(len(losing_trades)),
            "avgWin": float(avg_win),
            "avgLoss": float(avg_loss),
            "profitFactor": float(profit_factor),
            "currentATR": float(current_atr) if not np.isnan(current_atr) else 0.0,
            "currentVol": float(current_vol) if not np.isnan(current_vol) else 0.0
        })
    
    async def run_backtest_streaming(self, websocket, csv_path):
        """Run BOCS backtest with real-time streaming"""
        try:
            await self.send_log("🚀 Initializing BOCS Adaptive Strategy...")
            
            # Load data
            await self.send_log(f"📁 Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            await self.send_log(f"✓ Loaded {len(df)} bars")
            await self.send_log(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Initialize strategy
            initial_capital = 50000
            point_value = 1000
            
            self.strategy = BOCSStrategy(
                enable_session_filter=True,
                allowed_sessions=['London', 'NY AM'],
                overlap=False,
                strong=True,
                length_=100,
                length=14,
                atr_length=14,
                tp1_multiplier=2.0,
                tp2_multiplier=3.0,
                sl_multiplier=1.0,
                show_tp2=False
            )
            
            await self.send_log("⚙️ Strategy initialized with parameters:")
            await self.send_log(f"   • Session Filter: London, NY AM")
            await self.send_log(f"   • Length: 100, ATR: 14")
            await self.send_log(f"   • TP1: 2.0x, TP2: 3.0x, SL: 1.0x ATR")
            
            # Prepare data
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.sort_values('DateTime').reset_index(drop=True)
            
            # Calculate ATR
            df['atr'] = self.strategy.calculate_atr(df, self.strategy.atr_length)
            df['vol_sma'] = df['Volume'].rolling(window=20).mean()
            
            # Detect channels
            df = self.strategy.detect_channels(df)
            
            await self.send_log("📊 Starting backtest execution...")
            
            # Track state
            equity = initial_capital
            channel_start_idx = None
            all_trades = []
            
            # Stream data bar by bar
            for idx in range(self.strategy.length + 1, len(df)):
                row = df.iloc[idx]
                
                # Send candlestick
                await self.send_candlestick(row)
                
                # Send indicators
                if not pd.isna(row['atr']):
                    await self.send_indicator("ATR", row['DateTime'], row['atr'], "#00bcd4")
                
                if not pd.isna(row['vol']):
                    await self.send_indicator("VOL", row['DateTime'], row['vol'], "#ff9800")
                
                if not pd.isna(row['upper']):
                    await self.send_indicator("Upper", row['DateTime'], row['upper'], "#9c27b0")
                
                if not pd.isna(row['lower']):
                    await self.send_indicator("Lower", row['DateTime'], row['lower'], "#ff5722")
                
                # Check for exits if position exists
                if self.strategy.current_position is not None:
                    exit_type, exit_price = self.strategy.check_exit_conditions(idx, row)

                    # check_exit_conditions may close the position → re-check
                    if exit_type:
                        direction = (
                            self.strategy.current_position['direction']
                            if self.strategy.current_position and 'direction' in self.strategy.current_position
                            else "UNKNOWN"
                        )
                        await self.send_log(f"💰 {exit_type} hit! {direction} exit at {exit_price:.2f}")

                
                # Skip if session not allowed
                if not self.strategy.check_session_allowed(row['Session']):
                    await asyncio.sleep(0.01)  # Small delay
                    continue
                
                # Detect channel formation
                if df['lower_cross_upper'].iloc[idx]:
                    last_cross_idx = None
                    for i in range(idx-1, max(0, idx-self.strategy.length*2), -1):
                        if df['upper_cross_lower'].iloc[i]:
                            last_cross_idx = i
                            break
                    
                    if last_cross_idx is not None:
                        duration = idx - last_cross_idx
                        
                        if duration > 10:
                            h, l, vola = self.strategy.calculate_channel_bounds(df, last_cross_idx, idx)
                            
                            # Send channel box
                            await self.send_channel_box(
                                df.iloc[last_cross_idx]['DateTime'],
                                row['DateTime'],
                                h + vola,
                                l - vola,
                                "#8b5cf6"
                            )
                            
                            await self.send_log(f"📦 Channel formed: High={h:.2f}, Low={l:.2f}, Duration={duration} bars")
                            
                            channel = {
                                'start_idx': last_cross_idx,
                                'end_idx': idx,
                                'high': h,
                                'low': l,
                                'upper_bound': h,
                                'lower_bound': l,
                                'vola': vola,
                                'active': True
                            }
                            
                            if self.strategy.overlap or len(self.strategy.active_channels) == 0:
                                self.strategy.active_channels.append(channel)
                            else:
                                self.strategy.active_channels = [channel]
                
                # Check for breakouts
                if len(self.strategy.active_channels) > 0 and self.strategy.current_position is None:
                    for channel in self.strategy.active_channels[:]:
                        if not channel['active']:
                            continue
                        
                        close_price = row['Close']
                        open_price = row['Open']
                        
                        # Bullish breakout
                        if self.strategy.check_breakout(close_price, open_price, channel['upper_bound'], channel['lower_bound']):
                            if (self.strategy.strong and (close_price + open_price) / 2 > channel['upper_bound']) or \
                               (not self.strategy.strong and close_price > channel['upper_bound']):
                                
                                entry_price = channel['upper_bound']
                                atr = row['atr']
                                
                                if not np.isnan(atr):
                                    tp1, tp2, sl = self.strategy.calculate_tp_sl(entry_price, 'LONG', atr)
                                    self.strategy.update_position(idx, entry_price, 'LONG', tp1, tp2, sl, atr, row['DateTime'])
                                    
                                    # Send signal
                                    await self.send_signal(row, "BUY", entry_price, tp1, tp2, sl)
                                    await self.send_log(f"🔵 LONG entry at {entry_price:.2f} | TP1: {tp1:.2f}, SL: {sl:.2f}")
                                    
                                    channel['active'] = False
                                    self.strategy.active_channels.remove(channel)
                                    break
                            
                            # Bearish breakout
                            elif (self.strategy.strong and (close_price + open_price) / 2 < channel['lower_bound']) or \
                                 (not self.strategy.strong and close_price < channel['lower_bound']):
                                
                                entry_price = channel['lower_bound']
                                atr = row['atr']
                                
                                if not np.isnan(atr):
                                    tp1, tp2, sl = self.strategy.calculate_tp_sl(entry_price, 'SHORT', atr)
                                    self.strategy.update_position(idx, entry_price, 'SHORT', tp1, tp2, sl, atr, row['DateTime'])
                                    
                                    # Send signal
                                    await self.send_signal(row, "SELL", entry_price, tp1, tp2, sl)
                                    await self.send_log(f"🔴 SHORT entry at {entry_price:.2f} | TP1: {tp1:.2f}, SL: {sl:.2f}")
                                    
                                    channel['active'] = False
                                    self.strategy.active_channels.remove(channel)
                                    break
                        
                        channel['end_idx'] = idx
                
                # Update equity if trades exist
                if len(self.strategy.trades) > 0:
                    total_pnl_points = sum([t['pnl'] * t['qty'] for t in self.strategy.trades])
                    equity = initial_capital + (total_pnl_points * point_value)
                    all_trades = self.strategy.trades.copy()
                
                # Send metrics every 10 bars
                if idx % 10 == 0:
                    current_atr = row['atr'] if not pd.isna(row['atr']) else 0
                    current_vol = row['vol'] if not pd.isna(row['vol']) else 0
                    await self.send_metrics(equity, all_trades, initial_capital, current_atr, current_vol)
                
                # Delay for visualization (adjust speed here)
                await asyncio.sleep(0.05)  # 50ms delay = 20 bars per second
            
            # Final metrics
            await self.send_log(f"✅ Backtest complete! Total trades: {len(self.strategy.trades)}")
            
            if len(self.strategy.trades) > 0:
                current_atr = df.iloc[-1]['atr']
                current_vol = df.iloc[-1]['vol']
                await self.send_metrics(equity, self.strategy.trades, initial_capital, current_atr, current_vol)
                
                # Calculate final stats
                trades_df = pd.DataFrame(self.strategy.trades)
                trades_df['pnl_dollars'] = trades_df['pnl'] * point_value
                total_pnl = trades_df['pnl_dollars'].sum()
                win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
                
                await self.send_log(f"📈 Final Equity: ${equity:,.2f}")
                await self.send_log(f"💵 Total P&L: ${total_pnl:,.2f}")
                await self.send_log(f"🎯 Win Rate: {win_rate:.1f}%")
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            await self.send_log(error_msg, "ERROR")
            print(f"❌ Error in backtest: {e}")
            import traceback
            traceback.print_exc()
    
    async def handler(self, websocket):
        """Handle WebSocket connections"""
        await self.register(websocket)
        try:
            # Start backtest when client connects
            # Change this path to your actual CSV file
            csv_path = r"D:\Data\Session_V4.csv"
            
            await self.run_backtest_streaming(websocket, csv_path)
            
            # Keep connection alive
            async for message in websocket:
                pass
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)


async def main():
    server = BOCSWebSocketServer()
    
    print("\n" + "="*70)
    print("  BOCS ADAPTIVE STRATEGY - WEBSOCKET SERVER")
    print("="*70)
    print(f"\n  🌐 Server: ws://localhost:8000")
    print(f"  📊 Frontend: http://localhost:3000")
    print(f"\n  Waiting for frontend connection...")
    print("="*70 + "\n")
    
    async with websockets.serve(server.handler, "localhost", 8000):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Server stopped by user")