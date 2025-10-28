"use client"

import { useState } from "react"

interface UploadProgress {
  fileName: string
  progress: number
  status: "idle" | "uploading" | "success" | "error"
  error?: string
}

export function useFileUpload() {
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({
    fileName: "",
    progress: 0,
    status: "idle",
  })

  const uploadOHLCV = async (file: File) => {
    setUploadProgress({
      fileName: file.name,
      progress: 0,
      status: "uploading",
    })

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/upload/ohlcv", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || "Failed to upload OHLCV data")
      }

      const result = await response.json()

      setUploadProgress({
        fileName: file.name,
        progress: 100,
        status: "success",
      })

      console.log(`OHLCV data uploaded successfully: ${result.dataCount} records`)
      return result.data
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error"
      setUploadProgress({
        fileName: file.name,
        progress: 0,
        status: "error",
        error: errorMessage,
      })
      console.error("Error uploading OHLCV:", error)
      throw error
    }
  }

  const uploadNotebook = async (file: File) => {
    setUploadProgress({
      fileName: file.name,
      progress: 0,
      status: "uploading",
    })

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("/api/upload/notebook", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || "Failed to upload notebook")
      }

      const result = await response.json()

      setUploadProgress({
        fileName: file.name,
        progress: 100,
        status: "success",
      })

      console.log(`Notebook uploaded successfully: ${result.cellCount} code cells`)
      return result.cells
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error"
      setUploadProgress({
        fileName: file.name,
        progress: 0,
        status: "error",
        error: errorMessage,
      })
      console.error("Error uploading notebook:", error)
      throw error
    }
  }

  return {
    uploadOHLCV,
    uploadNotebook,
    uploadProgress,
  }
}
