$ErrorActionPreference = "Stop"
$jsonPath = Join-Path $PSScriptRoot "slides.json"
$outPath = Join-Path (Split-Path $PSScriptRoot -Parent) "AuroraBP_reproduction_report.pptx"
$raw = [System.IO.File]::ReadAllText($jsonPath, [System.Text.UTF8Encoding]::new($false))
$slides = $raw | ConvertFrom-Json

function To-Rgb([int]$r, [int]$g, [int]$b) { return $r + ($g * 256) + ($b * 65536) }
function To-Rtl([string]$text) { return ([char]0x202B) + $text + ([char]0x202C) }

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Add()
$pres.PageSetup.SlideWidth = 960
$pres.PageSetup.SlideHeight = 540

$index = 1
foreach ($item in $slides) {
    $slide = $pres.Slides.Add($index, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.Solid()
    $slide.Background.Fill.ForeColor.RGB = (To-Rgb 244 240 232)

    $bar = $slide.Shapes.AddShape(1, 0, 0, 960, 10)
    $bar.Fill.Solid()
    $bar.Fill.ForeColor.RGB = (To-Rgb 14 107 107)
    $bar.Line.Visible = 0

    $title = $slide.Shapes.AddTextbox(1, 28, 16, 900, 42)
    $title.TextFrame.WordWrap = -1
    $tr = $title.TextFrame.TextRange
    $tr.Text = To-Rtl ([string]$item.title)
    $tr.Font.Name = "Tahoma"
    $tr.Font.NameComplexScript = "Tahoma"
    $tr.Font.Size = 26
    $tr.Font.Bold = -1
    $tr.Font.Color.RGB = (To-Rgb 20 48 42)
    $tr.ParagraphFormat.Alignment = 3

    if ($item.bullets) {
        $box = $slide.Shapes.AddTextbox(1, 36, 72, 890, 420)
        $box.TextFrame.WordWrap = -1
        $body = $box.TextFrame.TextRange
        $body.Text = (($item.bullets | ForEach-Object { To-Rtl ([string]$_) }) -join "`r")
        $body.Font.Name = "Tahoma"
        $body.Font.NameComplexScript = "Tahoma"
        $body.Font.Size = 16
        $body.Font.Color.RGB = (To-Rgb 28 40 52)
        $body.ParagraphFormat.Alignment = 3
        $body.ParagraphFormat.Bullet.Type = 1
        $body.ParagraphFormat.Bullet.Visible = -1
        $body.ParagraphFormat.SpaceAfter = 8
    }

    if ($item.cards) {
        $cardIndex = 0
        foreach ($card in $item.cards) {
            $col = $cardIndex % 2
            $row = [math]::Floor($cardIndex / 2)
            $left = 28 + ($col * 460)
            $top = 78 + ($row * 200)
            $shape = $slide.Shapes.AddShape(1, $left, $top, 440, 180)
            $shape.Fill.Solid()
            $shape.Fill.ForeColor.RGB = (To-Rgb 255 255 255)
            $shape.Line.ForeColor.RGB = (To-Rgb 14 107 107)
            $range = $shape.TextFrame.TextRange
            $shape.TextFrame.WordWrap = -1
            $range.Text = (To-Rtl ([string]$card.head)) + "`r" + (To-Rtl ([string]$card.body))
            $range.Font.Name = "Tahoma"
            $range.Font.NameComplexScript = "Tahoma"
            $range.Font.Size = 14
            $range.Font.Color.RGB = (To-Rgb 28 40 52)
            $range.ParagraphFormat.Alignment = 3
            $range.Paragraphs(1).Font.Bold = -1
            $range.Paragraphs(1).Font.Color.RGB = (To-Rgb 14 107 107)
            $range.Paragraphs(1).Font.Size = 16
            $cardIndex++
        }
    }

    if ($item.headers) {
        $headers = @($item.headers)
        $rows = @($item.rows)
        $rowCount = $rows.Count + 1
        $colCount = $headers.Count
        $tableTop = 78
        if ($item.tableTop) { $tableTop = [double]$item.tableTop }
        $tableShape = $slide.Shapes.AddTable($rowCount, $colCount, 24, $tableTop, 912, 28 * $rowCount)
        $table = $tableShape.Table
        for ($c = 1; $c -le $colCount; $c++) {
            $cell = $table.Cell(1, $c)
            $cell.Shape.Fill.Solid()
            $cell.Shape.Fill.ForeColor.RGB = (To-Rgb 14 107 107)
            $text = $cell.Shape.TextFrame.TextRange
            $text.Text = To-Rtl ([string]$headers[$c - 1])
            $text.Font.Name = "Tahoma"
            $text.Font.NameComplexScript = "Tahoma"
            $text.Font.Size = 11
            $text.Font.Bold = -1
            $text.Font.Color.RGB = (To-Rgb 255 255 255)
            $text.ParagraphFormat.Alignment = 3
        }
        for ($r = 0; $r -lt $rows.Count; $r++) {
            $values = @($rows[$r])
            for ($c = 1; $c -le $colCount; $c++) {
                $cell = $table.Cell($r + 2, $c)
                $cell.Shape.Fill.Solid()
                if ($r % 2 -eq 0) { $cell.Shape.Fill.ForeColor.RGB = (To-Rgb 255 255 255) }
                else { $cell.Shape.Fill.ForeColor.RGB = (To-Rgb 231 242 242) }
                $text = $cell.Shape.TextFrame.TextRange
                $text.Text = To-Rtl ([string]$values[$c - 1])
                $text.Font.Name = "Tahoma"
                $text.Font.NameComplexScript = "Tahoma"
                $text.Font.Size = 11
                $text.Font.Color.RGB = (To-Rgb 28 40 52)
                $text.ParagraphFormat.Alignment = 3
            }
        }
    }

    if ($item.notes) {
        $noteTop = 400
        if ($item.noteTop) { $noteTop = [double]$item.noteTop }
        $note = $slide.Shapes.AddTextbox(1, 28, $noteTop, 900, 110)
        $note.TextFrame.WordWrap = -1
        $nr = $note.TextFrame.TextRange
        $nr.Text = (($item.notes | ForEach-Object { To-Rtl ([string]$_) }) -join "`r")
        $nr.Font.Name = "Tahoma"
        $nr.Font.NameComplexScript = "Tahoma"
        $nr.Font.Size = 13
        $nr.Font.Color.RGB = (To-Rgb 28 40 52)
        $nr.ParagraphFormat.Alignment = 3
        $nr.ParagraphFormat.Bullet.Type = 1
        $nr.ParagraphFormat.Bullet.Visible = -1
    }

    $foot = $slide.Shapes.AddTextbox(1, 28, 508, 760, 22)
    $fr = $foot.TextFrame.TextRange
    $fr.Text = To-Rtl "بازتولید Aurora-BP  ·  Mieloszyk و همکاران، IEEE JBHI 2022"
    $fr.Font.Name = "Tahoma"
    $fr.Font.NameComplexScript = "Tahoma"
    $fr.Font.Size = 11
    $fr.Font.Color.RGB = (To-Rgb 107 114 128)
    $fr.ParagraphFormat.Alignment = 3

    $page = $slide.Shapes.AddTextbox(1, 820, 508, 110, 22)
    $pr = $page.TextFrame.TextRange
    $pr.Text = "$index / $($slides.Count)"
    $pr.Font.Name = "Tahoma"
    $pr.Font.Size = 11
    $pr.Font.Color.RGB = (To-Rgb 107 114 128)
    $pr.ParagraphFormat.Alignment = 1
    $index++
}

$target = $outPath
if (Test-Path -LiteralPath $outPath) {
    try {
        Remove-Item -LiteralPath $outPath -Force -ErrorAction Stop
    } catch {
        $target = Join-Path (Split-Path $outPath -Parent) "AuroraBP_reproduction_report.updated.pptx"
    }
}
$pres.SaveAs($target)
$outPath = $target
$pres.Close()
$ppt.Quit()
Write-Output $outPath
