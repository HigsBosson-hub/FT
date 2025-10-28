"use client"

import { useEffect, useRef, useState, useCallback } from "react"

interface WebSocketMessage {
  type: string
  payload: any
}

interface UseWebSocketProps {
  onMessage: (data: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
}

export function useWebSocket({ onMessage, onConnect, onDisconnect, onError }: UseWebSocketProps) {
  const wsRef = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const messageQueueRef = useRef<WebSocketMessage[]>([])
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const MAX_RECONNECT_ATTEMPTS = 5
  const RECONNECT_DELAY = 3000

  const isValidWebSocketUrl = (url: string): boolean => {
    try {
      const urlObj = new URL(url)
      return urlObj.protocol === "ws:" || urlObj.protocol === "wss:"
    } catch {
      return false
    }
  }

  const connect = useCallback(
    (url: string) => {
      try {
        if (!isValidWebSocketUrl(url)) {
          const error = new Error(
            `Invalid WebSocket URL: "${url}". Must be a valid ws:// or wss:// URL. Example: ws://localhost:8000/ws`,
          )
          console.error("[WebSocket] Validation Error:", error.message)
          onError?.(error)
          setIsConnected(false)
          return
        }

        // Clear any pending reconnect attempts
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
        }

        console.log("[WebSocket] Attempting to connect to:", url)
        wsRef.current = new WebSocket(url)

        wsRef.current.onopen = () => {
          console.log("[WebSocket] Connected to", url)
          setIsConnected(true)
          setReconnectAttempts(0)
          onConnect?.()

          while (messageQueueRef.current.length > 0) {
            const message = messageQueueRef.current.shift()
            if (message && wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify(message))
            }
          }
        }

        wsRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as WebSocketMessage
            onMessage(data)
          } catch (error) {
            console.error("[WebSocket] Failed to parse message:", error)
          }
        }

        wsRef.current.onerror = (event) => {
          const wsError = wsRef.current
          let errorMessage = "WebSocket connection error"

          // Provide more specific error messages based on connection state
          if (wsError?.readyState === WebSocket.CONNECTING) {
            errorMessage = `Failed to connect to ${url}. Check if the server is running and the URL is correct.`
          } else if (wsError?.readyState === WebSocket.CLOSED) {
            errorMessage = `Connection to ${url} was closed unexpectedly. The server may have rejected the connection.`
          }

          const error = new Error(errorMessage)
          console.error("[WebSocket] Error:", {
            message: errorMessage,
            url,
            readyState: wsError?.readyState,
            event,
          })
          onError?.(error)
        }

        wsRef.current.onclose = () => {
          console.log("[WebSocket] Disconnected")
          setIsConnected(false)
          onDisconnect?.()

          if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            const delay = RECONNECT_DELAY * Math.pow(2, reconnectAttempts)
            console.log(
              `[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})`,
            )
            setReconnectAttempts((prev) => prev + 1)

            reconnectTimeoutRef.current = setTimeout(() => {
              connect(url)
            }, delay)
          } else {
            console.error("[WebSocket] Max reconnection attempts reached. Please check your backend server.")
          }
        }
      } catch (error) {
        const err = error instanceof Error ? error : new Error("Failed to create WebSocket connection")
        console.error("[WebSocket] Connection failed:", err.message)
        onError?.(err)
        setIsConnected(false)
      }
    },
    [onMessage, onConnect, onDisconnect, onError, reconnectAttempts],
  )

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
    setReconnectAttempts(0)
  }, [])

  const send = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      messageQueueRef.current.push(message)
      console.log("[WebSocket] Message queued, waiting for connection")
    }
  }, [])

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    connect,
    disconnect,
    send,
    isConnected,
    reconnectAttempts,
  }
}
