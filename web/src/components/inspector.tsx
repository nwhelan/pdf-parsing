import * as React from "react"

import { api, type Block, type DiffResponse, type ParseResult, type ScoreResponse } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScoresPanel } from "@/components/scores-panel"

const TABS: [string, string][] = [
  ["scores", "Scores"],
  ["text", "Text"],
  ["markdown", "Markdown"],
  ["extraction", "Extraction"],
  ["tables", "Tables"],
  ["diff", "Diff"],
  ["json", "JSON"],
]

interface Props {
  docId: string
  page: number
  scores: ScoreResponse | null
  result: ParseResult | undefined
  compareResult: ParseResult | undefined
  leftKey: string | null
  rightKey: string | null
  selectedBlock: Block | null
}

export function Inspector({
  docId,
  page,
  scores,
  result,
  leftKey,
  rightKey,
  selectedBlock,
}: Props) {
  const [tab, setTab] = React.useState("scores")
  const [diff, setDiff] = React.useState<DiffResponse | null>(null)
  const [diffError, setDiffError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (tab !== "diff" || !leftKey || !rightKey || leftKey === rightKey) return
    setDiff(null)
    setDiffError(null)
    api
      .diff(docId, leftKey, rightKey)
      .then(setDiff)
      .catch((err: Error) => setDiffError(err.message))
  }, [tab, docId, leftKey, rightKey])

  const pageResult = result?.pages.find((p) => p.page_number === page)

  return (
    <Tabs value={tab} onValueChange={setTab} className="flex h-full min-h-0 flex-col gap-0">
      <div className="bg-card flex h-11 shrink-0 items-center border-b px-2">
        <TabsList className="h-8">
          {TABS.map(([value, label]) => (
            <TabsTrigger key={value} value={value} className="px-2.5 text-xs">
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <TabsContent value="scores" className="m-0">
          <ScoresPanel scores={scores} />
        </TabsContent>

        <TabsContent value="text" className="m-0 p-3">
          {selectedBlock && (
            <div className="mb-3">
              <Badge variant="secondary" className="mb-1">
                {selectedBlock.layer}/{selectedBlock.kind}
              </Badge>
              <pre className="bg-muted rounded-md p-2 font-mono text-[11px] whitespace-pre-wrap">
                {selectedBlock.text || "(no text)"}
              </pre>
              <Separator className="mt-3" />
            </div>
          )}
          <pre className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
            {pageResult?.text || "Run a parser to see its text for this page."}
          </pre>
        </TabsContent>

        <TabsContent value="markdown" className="m-0 p-3">
          {result?.markdown ? (
            <pre className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{result.markdown}</pre>
          ) : (
            <p className="text-muted-foreground text-sm">
              {result ? `${result.parser_name} does not emit Markdown.` : "No result selected."}
            </p>
          )}
        </TabsContent>

        <TabsContent value="extraction" className="m-0 p-3">
          {result?.extraction != null ? (
            <pre className="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
              {JSON.stringify(result.extraction, null, 2)}
            </pre>
          ) : (
            <p className="text-muted-foreground text-sm">
              No structured extraction. Set an extraction schema in the parser's options —
              <span className="font-mono text-xs"> extraction_schema</span> for vision models,
              <span className="font-mono text-xs"> document_annotation_schema</span> for Mistral OCR —
              and the same schema across parsers makes their answers comparable.
            </p>
          )}
        </TabsContent>

        <TabsContent value="tables" className="m-0 space-y-4 p-3">
          {(pageResult?.tables.length ?? 0) === 0 ? (
            <p className="text-muted-foreground text-sm">No tables found on this page.</p>
          ) : (
            pageResult!.tables.map((table, index) => {
              const grid: string[][] = Array.from({ length: table.n_rows }, () =>
                Array<string>(table.n_cols).fill("")
              )
              table.cells.forEach((cell) => {
                if (grid[cell.row]) grid[cell.row][cell.col] = cell.text
              })
              return (
                <div key={table.id} className="space-y-1">
                  <p className="text-muted-foreground text-xs">
                    table {index + 1} — {table.n_rows}×{table.n_cols}
                  </p>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow className="hover:bg-transparent">
                          {(grid[0] ?? []).map((cell, i) => (
                            <TableHead key={i} className="h-7 px-2 text-[11px]">
                              {cell}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {grid.slice(1).map((row, r) => (
                          <TableRow key={r}>
                            {row.map((cell, c) => (
                              <TableCell key={c} className="px-2 py-1 text-[11px]">
                                {cell}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )
            })
          )}
        </TabsContent>

        <TabsContent value="diff" className="m-0 p-3">
          {!leftKey || !rightKey || leftKey === rightKey ? (
            <p className="text-muted-foreground text-sm">
              Turn on compare mode and pick two different parsers to diff them.
            </p>
          ) : diffError ? (
            <p className="text-destructive text-sm">{diffError}</p>
          ) : !diff ? (
            <p className="text-muted-foreground text-sm">Diffing…</p>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge variant="outline" className="tabular-nums">
                  raw {diff.text_similarity.toFixed(3)}
                </Badge>
                <Badge variant="outline" className="tabular-nums">
                  lines {diff.line_similarity.toFixed(3)}
                </Badge>
                <Badge variant="outline" className="tabular-nums">
                  CER {diff.cer}
                </Badge>
              </div>
              {diff.hunks.length === 0 ? (
                <p className="text-muted-foreground text-xs leading-relaxed">
                  {diff.same_content_different_order
                    ? "Same lines recovered — these parsers differ in serialization order, not content."
                    : "Identical line by line."}
                </p>
              ) : (
                diff.hunks.map((hunk, i) => (
                  <div key={i} className="border-l-2 pl-2 font-mono text-[11px]">
                    {hunk.left.map((line, j) => (
                      <p key={`l${j}`} className="text-red-600 dark:text-red-400">
                        − {line}
                      </p>
                    ))}
                    {hunk.right.map((line, j) => (
                      <p key={`r${j}`} className="text-emerald-600 dark:text-emerald-400">
                        + {line}
                      </p>
                    ))}
                  </div>
                ))
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="json" className="m-0 p-3">
          <pre className="font-mono text-[10px] leading-relaxed whitespace-pre-wrap">
            {pageResult ? JSON.stringify(pageResult, null, 1) : "No result selected."}
          </pre>
        </TabsContent>
      </ScrollArea>
    </Tabs>
  )
}
