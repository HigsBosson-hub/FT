"""
BOCS Strategy WebSocket Backend
Connects your BOCS Adaptive strategy to the frontend visualizer
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BOCSBacktestServer:
    def __init__(self):
        self.clients = set()
        self.df = None
        
    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
    
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
        """Send log message"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] {level}: {message}"
        await self.send_message("log", log_msg)
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    def calculate_volatility(self, df, length=100, vol_period=14):
        """Calculate normalized volatility as in BOCS strategy"""
        if len(df) < length:
            return 0
        
        # Normalized price
        high_max = df['High'].rolling(window=length).max()
        low_min = df['Low'].rolling(window=length).min()
        
        normalized = (df['Close'] - low_min) / (high_max - low_min)
        vol = normalized.rolling(window=vol_period).std()
        
        return vol.iloc[-1] if len(vol) > 0 and not pd.isna(vol.iloc[-1]) else 0
    
    def detect_channel_signals(self, df, i, length=14):
        """Detect upper/lower crossover signals for channel formation"""
        if i < length + 14:
            return None, None, None
        
        # Calculate vol and bounds
        vol_series = []
        for j in range(i - length - 14, i):
            vol = self.calculate_volatility(df.iloc[:j+1], length_=100, vol_period=14)
            vol_series.append(vol)
        
        vol_df = pd.Series(vol_series)
        upper = vol_df.rolling(window=length).max().iloc[-1]
        lower = vol_df.rolling(window=length).min().iloc[-1]
        current_vol = vol_series[-1] if vol_series else 0
        
        # Detect crossovers
        upper_cross = current_vol >= upper if i > 0 else False
        lower_cross = current_vol <= lower if i > 0 else False
        
        return upper_cross, lower_cross, current_vol
    
    async def run_backtest(self, websocket):
        """Run BOCS backtest and stream results"""
        try:
            await self.send_log("Initializing BOCS Adaptive Strategy...")
            
            # Load your data here
            # For demo purposes, generating sample data
            await self.send_log("Loading historical data...")
            
            # Generate sample OHLCV data (replace with your actual data loading)
            dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
            np.random.seed(42)
            
            base_price = 4500
            data = []
            for i, date in enumerate(dates):
                change = np.random.randn() * 10
                base_price += change
                
                open_price = base_price
                high_price = base_price + abs(np.random.randn() * 15)
                low_price = base_price - abs(np.random.randn() * 15)
                close_price = base_price + np.random.randn() * 8
                volume = np.random.randint(1000, 5000)
                
                data.append({
                    'DateTime': date,
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
            
            self.df = pd.DataFrame(data)
            await self.send_log(f"Loaded {len(self.df)} candles")
            
            # Initialize strategy variables
            equity = 100000
            initial_equity = equity
            position = None
            position_size = 1
            trades = []
            in_channel = False
            channel_start_idx = 0
            
            await self.send_log("Starting backtest execution...")
            
            # Main backtest loop
            for i in range(100, len(self.df)):  # Start after warmup period
                row = self.df.iloc[i]
                
                # Send candlestick data
                await self.send_message("candlestick", {
                    "time": int(row['DateTime'].timestamp()),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": float(row['Volume'])
                })
                
                # Calculate indicators
                atr = self.calculate_atr(self.df.iloc[max(0, i-14):i+1], period=14)
                vol = self.calculate_volatility(self.df.iloc[max(0, i-100):i+1], length=100, vol_period=14)
                
                # Send indicator updates
                await self.send_message("indicator", {
                    "name": "ATR",
                    "values": [{"time": int(row['DateTime'].timestamp()), "value": float(atr)}],
                    "color": "#00bcd4"
                })
                
                await self.send_message("indicator", {
                    "name": "VOL",
                    "values": [{"time": int(row['DateTime'].timestamp()), "value": float(vol)}],
                    "color": "#ff9800"
                })
                
                # Detect channel formation
                upper_cross, lower_cross, current_vol = self.detect_channel_signals(self.df, i)
                
                if upper_cross or lower_cross:
                    if not in_channel:
                        in_channel = True
                        channel_start_idx = i
                        channel_type = "upper_cross_lower" if upper_cross else "lower_cross_upper"
                        await self.send_log(f"Channel formed: {channel_type}")
                
                # Check for breakout signals
                if in_channel and i > channel_start_idx + 5:
                    channel_high = self.df.iloc[channel_start_idx:i]['High'].max()
                    channel_low = self.df.iloc[channel_start_idx:i]['Low'].min()
                    
                    # Send channel visualization
                    if i - channel_start_idx > 10:
                        await self.send_message("box", {
                            "startTime": int(self.df.iloc[channel_start_idx]['DateTime'].timestamp()),
                            "endTime": int(row['DateTime'].timestamp()),
                            "topPrice": float(channel_high + atr * 0.5),
                            "bottomPrice": float(channel_low - atr * 0.5),
                            "color": "#8b5cf6"
                        })
                    
                    # Long breakout
                    if position is None and row['Close'] > channel_high:
                        position = 'LONG'
                        entry_price = row['Close']
                        tp1 = entry_price + atr * 1.5
                        tp2 = entry_price + atr * 3.0
                        sl = entry_price - atr * 1.0
                        
                        await self.send_message("signal", {
                            "time": int(row['DateTime'].timestamp()),
                            "type": "BUY",
                            "price": float(entry_price),
                            "tp1": float(tp1),
                            "tp2": float(tp2),
                            "sl": float(sl)
                        })
                        
                        await self.send_log(f"LONG entry at {entry_price:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}, SL: {sl:.2f}")
                        
                        trades.append({
                            'entry_time': row['DateTime'],
                            'entry_price': entry_price,
                            'type': 'LONG',
                            'tp1': tp1,
                            'tp2': tp2,
                            'sl': sl
                        })
                        
                        in_channel = False
                    
                    # Short breakout
                    elif position is None and row['Close'] < channel_low:
                        position = 'SHORT'
                        entry_price = row['Close']
                        tp1 = entry_price - atr * 1.5
                        tp2 = entry_price - atr * 3.0
                        sl = entry_price + atr * 1.0
                        
                        await self.send_message("signal", {
                            "time": int(row['DateTime'].timestamp()),
                            "type": "SELL",
                            "price": float(entry_price),
                            "tp1": float(tp1),
                            "tp2": float(tp2),
                            "sl": float(sl)
                        })
                        
                        await self.send_log(f"SHORT entry at {entry_price:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}, SL: {sl:.2f}")
                        
                        trades.append({
                            'entry_time': row['DateTime'],
                            'entry_price': entry_price,
                            'type': 'SHORT',
                            'tp1': tp1,
                            'tp2': tp2,
                            'sl': sl
                        })
                        
                        in_channel = False
                
                # Check for exit conditions
                if position and len(trades) > 0:
                    last_trade = trades[-1]
                    pnl = 0
                    
                    if position == 'LONG':
                        if row['High'] >= last_trade['tp1']:
                            pnl = (last_trade['tp1'] - last_trade['entry_price']) * position_size * 1000
                            equity += pnl
                            position = None
                            await self.send_log(f"TP1 hit! P&L: ${pnl:.2f}")
                        elif row['Low'] <= last_trade['sl']:
                            pnl = (last_trade['sl'] - last_trade['entry_price']) * position_size * 1000
                            equity += pnl
                            position = None
                            await self.send_log(f"Stop loss hit. P&L: ${pnl:.2f}")
                    
                    elif position == 'SHORT':
                        if row['Low'] <= last_trade['tp1']:
                            pnl = (last_trade['entry_price'] - last_trade['tp1']) * position_size * 1000
                            equity += pnl
                            position = None
                            await self.send_log(f"TP1 hit! P&L: ${pnl:.2f}")
                        elif row['High'] >= last_trade['sl']:
                            pnl = (last_trade['entry_price'] - last_trade['sl']) * position_size * 1000
                            equity += pnl
                            position = None
                            await self.send_log(f"Stop loss hit. P&L: ${pnl:.2f}")
                    
                    if pnl != 0:
                        trades[-1]['exit_price'] = row['Close']
                        trades[-1]['pnl'] = pnl
                        trades[-1]['exit_time'] = row['DateTime']
                
                # Calculate and send metrics every 10 candles
                if i % 10 == 0:
                    winning_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
                    losing_trades = [t for t in trades if 'pnl' in t and t['pnl'] < 0]
                    
                    total_trades = len([t for t in trades if 'pnl' in t])
                    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                    
                    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                    
                    max_dd = max(0, (initial_equity - equity) / initial_equity)
                    current_dd = max(0, (initial_equity - equity) / initial_equity)
                    
                    profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
                    
                    await self.send_message("metrics", {
                        "equity": float(equity),
                        "pnl": float(equity - initial_equity),
                        "drawdown": float(current_dd),
                        "winRate": float(win_rate),
                        "trades": int(total_trades),
                        "sharpeRatio": float(1.5),  # Calculate properly in your implementation
                        "maxDrawdown": float(max_dd),
                        "totalWins": len(winning_trades),
                        "totalLosses": len(losing_trades),
                        "avgWin": float(avg_win),
                        "avgLoss": float(avg_loss),
                        "profitFactor": float(profit_factor),
                        "currentATR": float(atr),
                        "currentVol": float(vol)
                    })
                
                # Small delay for visualization
                await asyncio.sleep(0.05)
            
            await self.send_log("Backtest completed successfully")
            
        except Exception as e:
            await self.send_log(f"ERROR: {str(e)}", "ERROR")
            print(f"Error in backtest: {e}")
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        await self.register(websocket)
        try:
            # Start backtest when client connects
            await self.run_backtest(websocket)
            
            # Keep connection alive
            async for message in websocket:
                # Handle any incoming messages from client if needed
                pass
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

async def main():
    server = BOCSBacktestServer()
    
    print("=" * 60)
    print("BOCS Strategy WebSocket Server")
    print("=" * 60)
    print("Server starting on ws://localhost:8000")
    print("Waiting for frontend connection...")
    print("=" * 60)
    
    async with websockets.serve(server.handler, "localhost", 8000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")