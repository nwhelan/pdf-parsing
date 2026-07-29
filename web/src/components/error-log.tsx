import * as React from "react"
import { Check, ChevronRight, Copy, Lightbulb, TriangleAlert } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { cn } from "@/lib/utils"

export interface RunError {
  id: string
  parserId: string
  parserName: string
  at: string
  message: string
  /** What the adapter sent before it failed, redacted server-side. */
  debug: unknown[]
  /** The adapter's traceback, when the failure came from inside the run. */
  detail: string[]
}

interface Props {
  errors: RunError[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onClear: () => void
}

function CopyButton({ value, label = "Copy" }: { value: unknown; label?: string }) {
  const [copied, setCopied] = React.useState(false)
  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-6 gap-1 px-2 text-xs"
      onClick={() => {
        const text = typeof value === "string" ? value : JSON.stringify(value, null, 2)
        void navigator.clipboard?.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1200)
      }}
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Copied" : label}
    </Button>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0 space-y-1">
      <p className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">{title}</p>
      {children}
    </div>
  )
}

/** One failure: the message, then the request that produced it. */
function ErrorEntry({ error, defaultOpen }: { error: RunError; defaultOpen: boolean }) {
  const [open, setOpen] = React.useState(defaultOpen)

  // Adapters attach these to a recorded request when the configuration looks
  // like something the API will refuse.
  const hints = error.debug.flatMap((event) =>
    Array.isArray((event as { hints?: unknown }).hints) ? ((event as { hints: string[] }).hints) : []
  )

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="min-w-0 rounded-md border">
      <CollapsibleTrigger className="hover:bg-muted/50 flex w-full items-start gap-2 rounded-md p-2.5 text-left">
        <ChevronRight
          className={cn("text-muted-foreground mt-0.5 size-3.5 shrink-0 transition-transform", open && "rotate-90")}
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">{error.parserName}</span>
            <span className="text-muted-foreground text-[11px]">{error.at}</span>
          </div>
          <p className={cn("text-destructive text-xs", !open && "truncate")}>{error.message}</p>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent className="min-w-0 space-y-3 border-t p-2.5">
        {/* Hints first: the API's own message rarely says which convention it
            expected, and this is the part you can act on. */}
        {hints.length > 0 && (
          <div className="min-w-0 space-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 p-2">
            {hints.map((hint, i) => (
              <p key={i} className="flex min-w-0 gap-1.5 text-xs break-words">
                <Lightbulb className="mt-0.5 size-3 shrink-0 text-amber-600 dark:text-amber-400" />
                <span>{hint}</span>
              </p>
            ))}
          </div>
        )}

        <Block title="Error">
          <pre className="bg-muted text-destructive overflow-x-auto rounded p-2 font-mono text-[11px] whitespace-pre-wrap">
            {error.message}
          </pre>
        </Block>

        {error.debug.length > 0 ? (
          <Block title={`Request sent (${error.debug.length} step${error.debug.length > 1 ? "s" : ""})`}>
            <pre className="bg-muted max-h-72 overflow-y-auto rounded p-2 font-mono text-[11px] break-words whitespace-pre-wrap">
              {JSON.stringify(error.debug, null, 2)}
            </pre>
          </Block>
        ) : (
          <p className="text-muted-foreground text-xs">
            No request was recorded — the run failed before anything was sent (a missing key or an
            unreachable endpoint). Turn on the parser's <span className="font-mono">debug</span> option
            for the full prompt and response body on the next run.
          </p>
        )}

        {error.detail.length > 0 && (
          <Block title="Traceback">
            <pre className="bg-muted text-muted-foreground max-h-48 overflow-y-auto rounded p-2 font-mono text-[10px] break-words whitespace-pre-wrap">
              {error.detail.join("\n")}
            </pre>
          </Block>
        )}

        <div className="flex justify-end">
          <CopyButton value={error} label="Copy all" />
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

/** The button that lives in the header; renders nothing until something fails. */
export function ErrorLogButton({
  errors,
  onOpen,
}: {
  errors: RunError[]
  onOpen: () => void
}) {
  if (errors.length === 0) return null
  return (
    <Button variant="ghost" size="sm" className="text-destructive h-7 gap-1.5 px-2" onClick={onOpen}>
      <TriangleAlert className="size-3.5" />
      <span className="text-xs">{errors.length}</span>
    </Button>
  )
}

export function ErrorLog({ errors, open, onOpenChange, onClear }: Props) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[80vh] overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <TriangleAlert className="text-destructive size-4" />
            Failures
            <Badge variant="secondary">{errors.length}</Badge>
          </DialogTitle>
          <DialogDescription>
            Every failure this session, newest first, with the request that preceded it.
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-[55vh] min-w-0 overflow-y-auto pr-3">
          <div className="min-w-0 space-y-2">
            {errors.length === 0 ? (
              <p className="text-muted-foreground py-6 text-center text-sm">Nothing has failed.</p>
            ) : (
              errors.map((error, index) => (
                <ErrorEntry key={error.id} error={error} defaultOpen={index === 0} />
              ))
            )}
          </div>
        </div>

        <div className="flex items-center justify-between border-t pt-3">
          <CopyButton value={errors} label="Copy all failures" />
          <Button variant="outline" size="sm" className="h-7 text-xs" onClick={onClear}>
            Clear
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
