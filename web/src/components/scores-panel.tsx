import { AlertTriangle, Banknote, Gauge, GitCompare, Target } from "lucide-react"

import type { ScoreResponse } from "@/lib/api"
import { cn } from "@/lib/utils"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

/** Green / amber / red for a 0-1 score, so a scan of the column reads fast. */
function tone(value: number | null | undefined, good = 0.95, ok = 0.8) {
  if (value == null) return "text-muted-foreground"
  if (value >= good) return "text-emerald-600 dark:text-emerald-400"
  if (value >= ok) return "text-amber-600 dark:text-amber-400"
  return "text-red-600 dark:text-red-400"
}

const num = (v: number | null | undefined, digits = 2) => (v == null ? "–" : v.toFixed(digits))

function SectionCard({
  title,
  icon,
  children,
}: {
  title: string
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-muted-foreground flex items-center gap-2 text-xs font-medium tracking-wide uppercase">
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4">{children}</CardContent>
    </Card>
  )
}

export function ScoresPanel({
  scores,
  onShowErrors,
}: {
  scores: ScoreResponse | null
  onShowErrors?: () => void
}) {
  if (!scores || scores.rows.length === 0) {
    return (
      <p className="text-muted-foreground p-4 text-sm">
        Run one or more parsers to see how they compare.
      </p>
    )
  }

  const ok = scores.rows.filter((r) => r.status === "ok")
  const failed = scores.rows.filter((r) => r.status !== "ok")
  const isBank = scores.doc_class === "bank_statement"
  const withLedger = ok.filter((r) => r.ledger_score)
  const similarityIds = Object.keys(scores.similarity)

  return (
    <div className="space-y-3 p-3">
      {/* One line, not a stack of full-width alerts. Failed results are stored,
          so the old rendering re-buried the scores on every reload. */}
      {failed.length > 0 && (
        <Alert variant="destructive" className="flex items-center gap-2 py-2">
          <AlertTriangle className="size-4" />
          <AlertDescription className="flex-1 truncate text-xs">
            {failed.length === 1
              ? `${failed[0].parser_id} failed: ${failed[0].error ?? ""}`
              : `${failed.length} parsers failed: ${failed.map((r) => r.parser_id).join(", ")}`}
          </AlertDescription>
          {onShowErrors && (
            <Button variant="outline" size="sm" className="h-6 shrink-0 px-2 text-xs" onClick={onShowErrors}>
              Details
            </Button>
          )}
        </Alert>
      )}

      <SectionCard title="Generic signals" icon={<Gauge className="size-3.5" />}>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="h-8 px-1 text-xs">parser</TableHead>
              <TableHead className="h-8 px-1 text-right text-xs">s/pg</TableHead>
              <TableHead className="h-8 px-1 text-right text-xs">chars</TableHead>
              <TableHead className="h-8 px-1 text-right text-xs">lines</TableHead>
              <TableHead className="h-8 px-1 text-right text-xs">cover</TableHead>
              <TableHead className="h-8 px-1 text-right text-xs">order</TableHead>
              <TableHead className="h-8 px-1 text-right text-xs">cost</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {ok.map((row) => (
              <TableRow key={row.key}>
                <TableCell className="px-1 py-1.5 text-xs font-medium">{row.parser_id}</TableCell>
                <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">
                  {num(row.seconds_per_page)}
                </TableCell>
                <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">{row.n_chars}</TableCell>
                <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">{row.n_lines}</TableCell>
                <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">
                  {num(row.page_coverage)}
                </TableCell>
                <TableCell
                  className={cn("px-1 py-1.5 text-right text-xs tabular-nums", tone(row.reading_order_score))}
                >
                  {num(row.reading_order_score)}
                </TableCell>
                <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">
                  {row.cost_usd != null ? `$${row.cost_usd.toFixed(4)}` : "–"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </SectionCard>

      {isBank && (
        <SectionCard title="Bank statement" icon={<Banknote className="size-3.5" />}>
          <p className="text-muted-foreground mb-3 text-[11px] leading-snug">
            A statement checks itself: each row's balance delta must equal its amount. Reconciliation needs
            no ground truth.
          </p>
          <div className="space-y-3">
            {ok.map((row) => {
              const b = row.bank_statement
              if (!b) return null
              return (
                <div key={row.key} className="space-y-1.5">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="font-medium">{row.parser_id}</span>
                    <div className="flex items-center gap-1.5">
                      <Badge variant="outline" className="tabular-nums">
                        {b.n_transactions} txns
                      </Badge>
                      {b.totals_match != null && (
                        <Badge variant={b.totals_match ? "secondary" : "destructive"}>
                          {b.totals_match ? "totals ✓" : "totals ✗"}
                        </Badge>
                      )}
                      <span className={cn("w-10 text-right tabular-nums", tone(b.reconciliation_rate, 0.95, 0.7))}>
                        {(b.reconciliation_rate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <Progress value={b.reconciliation_rate * 100} className="h-1.5" />
                  <div className="text-muted-foreground flex gap-3 text-[11px] tabular-nums">
                    <span>
                      amount col {num(b.amount_column_consistency)}
                      {b.amount_columns_detected > 1 ? ` (${b.amount_columns_detected} cols)` : ""}
                    </span>
                    <span>balance col {num(b.balance_column_consistency)}</span>
                  </div>
                </div>
              )
            })}
          </div>

          {ok.some((r) => (r.bank_statement?.breaks.length ?? 0) > 0) && (
            <div className="mt-4 space-y-2">
              <p className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
                Reconciliation breaks
              </p>
              {ok.flatMap((row) =>
                (row.bank_statement?.breaks ?? []).slice(0, 3).map((brk, i) => (
                  <div key={`${row.key}-${i}`} className="border-l-2 border-red-500/60 pl-2">
                    <p className="text-muted-foreground text-[11px]">
                      {row.parser_id} · page {brk.page} · expected Δ {brk.expected_delta}, read {brk.amount}
                    </p>
                    <p className="font-mono text-[11px] break-all">{brk.raw}</p>
                  </div>
                ))
              )}
            </div>
          )}
        </SectionCard>
      )}

      {withLedger.length > 0 && (
        <SectionCard title="Vs. ground truth" icon={<Target className="size-3.5" />}>
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="h-8 px-1 text-xs">parser</TableHead>
                <TableHead className="h-8 px-1 text-right text-xs">P</TableHead>
                <TableHead className="h-8 px-1 text-right text-xs">R</TableHead>
                <TableHead className="h-8 px-1 text-right text-xs">F1</TableHead>
                <TableHead className="h-8 px-1 text-right text-xs">desc</TableHead>
                <TableHead className="h-8 px-1 text-right text-xs">bal</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {withLedger.map((row) => {
                const s = row.ledger_score!
                return (
                  <TableRow key={row.key}>
                    <TableCell className="px-1 py-1.5 text-xs font-medium">{row.parser_id}</TableCell>
                    <TableCell className={cn("px-1 py-1.5 text-right text-xs tabular-nums", tone(s.precision))}>
                      {num(s.precision)}
                    </TableCell>
                    <TableCell className={cn("px-1 py-1.5 text-right text-xs tabular-nums", tone(s.recall))}>
                      {num(s.recall)}
                    </TableCell>
                    <TableCell className={cn("px-1 py-1.5 text-right text-xs tabular-nums", tone(s.f1))}>
                      {num(s.f1)}
                    </TableCell>
                    <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">
                      {num(s.description_similarity)}
                    </TableCell>
                    <TableCell className="px-1 py-1.5 text-right text-xs tabular-nums">
                      {num(s.balance_accuracy)}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </SectionCard>
      )}

      {similarityIds.length > 1 && (
        <SectionCard title="Text agreement" icon={<GitCompare className="size-3.5" />}>
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="h-8 px-1 text-xs" />
                {similarityIds.map((id) => (
                  <TableHead key={id} className="h-8 px-1 text-right text-xs">
                    {id.slice(0, 6)}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {similarityIds.map((a) => (
                <TableRow key={a}>
                  <TableCell className="px-1 py-1.5 text-xs font-medium">{a}</TableCell>
                  {similarityIds.map((b) => (
                    <TableCell
                      key={b}
                      className={cn(
                        "px-1 py-1.5 text-right text-xs tabular-nums",
                        a === b ? "text-muted-foreground" : tone(scores.similarity[a][b])
                      )}
                    >
                      {scores.similarity[a][b].toFixed(2)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </SectionCard>
      )}
    </div>
  )
}
