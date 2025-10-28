import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Validate file type
    const validTypes = ["text/csv", "application/json"]
    if (!validTypes.includes(file.type)) {
      return NextResponse.json({ error: "Invalid file type. Please upload CSV or JSON." }, { status: 400 })
    }

    // Read file content
    const content = await file.text()

    // Parse OHLCV data
    let ohlcvData: any[] = []

    if (file.type === "text/csv") {
      // Parse CSV
      const lines = content.split("\n").filter((line) => line.trim())
      const headers = lines[0].split(",").map((h) => h.trim().toLowerCase())

      ohlcvData = lines.slice(1).map((line) => {
        const values = line.split(",")
        return {
          time: Number.parseInt(headers.includes("time") ? values[headers.indexOf("time")] : values[0]),
          open: Number.parseFloat(headers.includes("open") ? values[headers.indexOf("open")] : values[1]),
          high: Number.parseFloat(headers.includes("high") ? values[headers.indexOf("high")] : values[2]),
          low: Number.parseFloat(headers.includes("low") ? values[headers.indexOf("low")] : values[3]),
          close: Number.parseFloat(headers.includes("close") ? values[headers.indexOf("close")] : values[4]),
          volume: Number.parseFloat(headers.includes("volume") ? values[headers.indexOf("volume")] : values[5] || "0"),
        }
      })
    } else if (file.type === "application/json") {
      // Parse JSON
      ohlcvData = JSON.parse(content)
    }

    return NextResponse.json({
      success: true,
      message: "OHLCV data uploaded successfully",
      dataCount: ohlcvData.length,
      data: ohlcvData,
    })
  } catch (error) {
    console.error("Error uploading OHLCV:", error)
    return NextResponse.json({ error: "Failed to process file" }, { status: 500 })
  }
}
