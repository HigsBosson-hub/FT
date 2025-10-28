import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Validate file type
    if (file.type !== "application/json" && !file.name.endsWith(".ipynb")) {
      return NextResponse.json(
        { error: "Invalid file type. Please upload a Jupyter notebook (.ipynb)." },
        { status: 400 },
      )
    }

    // Read file content
    const content = await file.text()
    const notebook = JSON.parse(content)

    // Extract code cells
    const codeCells = notebook.cells
      .filter((cell: any) => cell.cell_type === "code")
      .map((cell: any) => cell.source.join(""))

    return NextResponse.json({
      success: true,
      message: "Notebook uploaded successfully",
      cellCount: codeCells.length,
      cells: codeCells,
    })
  } catch (error) {
    console.error("Error uploading notebook:", error)
    return NextResponse.json({ error: "Failed to process notebook" }, { status: 500 })
  }
}
