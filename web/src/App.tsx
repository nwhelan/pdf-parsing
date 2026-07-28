import * as React from "react"
import {
  ChevronLeft,
  ChevronRight,
  Columns2,
  FileUp,
  Maximize2,
  Moon,
  Sun,
  Tags,
  ZoomIn,
} from "lucide-react"
import { toast } from "sonner"

import { api, type Block, type DocumentDetail, type DocumentMeta, type ParseResult, type ParserSpec, type Preset, type ScoreResponse } from "@/lib/api"
import { useTheme } from "@/components/theme-provider"
import { Inspector } from "@/components/inspector"
import { PageViewer } from "@/components/page-viewer"
import { ParserSidebar } from "@/components/parser-sidebar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Toaster } from "@/components/ui/sonner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

export default function App() {
  const { resolvedTheme, setTheme } = useTheme()

  const [parsers, setParsers] = React.useState<ParserSpec[]>([])
  const [parsersLoading, setParsersLoading] = React.useState(true)
  const [documents, setDocuments] = React.useState<DocumentMeta[]>([])
  const [doc, setDoc] = React.useState<DocumentDetail | null>(null)
  const [results, setResults] = React.useState<Record<string, ParseResult>>({})
  const [scores, setScores] = React.useState<ScoreResponse | null>(null)

  const [selected, setSelected] = React.useState<Set<string>>(new Set())
  const [configuring, setConfiguring] = React.useState<string | null>(null)
  const [optionValues, setOptionValues] = React.useState<Record<string, Record<string, unknown>>>({})
  const [presets, setPresets] = React.useState<Preset[]>([])
  const [force, setForce] = React.useState(false)
  const [running, setRunning] = React.useState(false)

  const [page, setPage] = React.useState(1)
  const [zoom, setZoom] = React.useState(1.4)
  const [fit, setFit] = React.useState(true)
  const [compare, setCompare] = React.useState(false)
  const [showLabels, setShowLabels] = React.useState(false)
  const [leftKey, setLeftKey] = React.useState<string | null>(null)
  const [rightKey, setRightKey] = React.useState<string | null>(null)
  const [leftLayers, setLeftLayers] = React.useState<string[]>(["line"])
  const [rightLayers, setRightLayers] = React.useState<string[]>(["line"])
  const [selectedBlock, setSelectedBlock] = React.useState<Block | null>(null)

  const fileInput = React.useRef<HTMLInputElement>(null)

  // -- loading ------------------------------------------------------------

  // The parser list is the whole left rail, so one failed fetch must not end
  // the session. `pdfplay serve` may still be binding its port when the page
  // opens, or the request may land while it restarts — both are transient, and
  // both used to leave the sidebar permanently empty.
  const loadParsers = React.useCallback(async function load(attempt = 0): Promise<void> {
    try {
      setParsers(await api.parsers())
      setParsersLoading(false)
    } catch (err) {
      if (attempt < 4) {
        await new Promise((resolve) => setTimeout(resolve, 400 * 2 ** attempt))
        return load(attempt + 1)
      }
      setParsersLoading(false)
      toast.error("Could not load parsers", {
        description: `${(err as Error).message} — check the terminal running \`pdfplay serve\`.`,
        action: { label: "Retry", onClick: () => void load() },
        duration: Infinity,
      })
    }
  }, [])

  React.useEffect(() => {
    void loadParsers()
    api.presets().then(setPresets).catch(() => setPresets([]))
    api
      .documents()
      .then((docs) => {
        setDocuments(docs)
        if (docs.length) void selectDocument(docs[0].doc_id)
      })
      .catch((err: Error) => toast.error("Could not load documents", { description: err.message }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const refreshScores = React.useCallback(
    async (docId: string, keys: string[]) => {
      if (!keys.length) return setScores(null)
      try {
        setScores(await api.score(docId, keys))
      } catch (err) {
        toast.error("Scoring failed", { description: (err as Error).message })
      }
    },
    []
  )

  const selectDocument = React.useCallback(
    async (docId: string) => {
      const detail = await api.document(docId)
      setDoc(detail)
      setPage(1)
      setSelectedBlock(null)

      const loaded: Record<string, ParseResult> = {}
      for (const meta of detail.results) {
        try {
          loaded[meta.key] = await api.result(docId, meta.key)
        } catch {
          /* skip unreadable results */
        }
      }
      setResults(loaded)
      const keys = Object.keys(loaded)
      setLeftKey(keys[0] ?? null)
      setRightKey(keys[1] ?? keys[0] ?? null)
      await refreshScores(docId, keys)
    },
    [refreshScores]
  )

  // -- actions ------------------------------------------------------------

  async function runSelected() {
    if (!doc) return
    setRunning(true)
    const next = { ...results }
    for (const id of selected) {
      const toastId = toast.loading(`Running ${id}…`)
      try {
        const out = await api.parse(doc.doc_id, id, { options: optionValues[id] ?? {}, force })
        next[out.key] = out.result
        if (out.result.status === "error") {
          toast.error(`${id} failed`, { id: toastId, description: out.result.error ?? "" })
        } else {
          toast.success(`${id} — ${out.result.duration_s.toFixed(2)}s${out.cached ? " (cached)" : ""}`, {
            id: toastId,
          })
        }
      } catch (err) {
        toast.error(`${id} failed`, { id: toastId, description: (err as Error).message })
      }
    }
    setResults(next)
    const keys = Object.keys(next)
    setLeftKey((k) => k ?? keys[0] ?? null)
    setRightKey((k) => k ?? keys[1] ?? keys[0] ?? null)
    await refreshScores(doc.doc_id, keys)
    setRunning(false)
  }

  async function upload(file: File) {
    const toastId = toast.loading(`Uploading ${file.name}…`)
    try {
      const meta = await api.upload(file)
      setDocuments(await api.documents())
      await selectDocument(meta.doc_id)
      toast.success(`Added ${meta.name}`, { id: toastId, description: `${meta.pages} page(s)` })
    } catch (err) {
      toast.error("Upload failed", { id: toastId, description: (err as Error).message })
    }
  }

  React.useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return
      if (event.key === "ArrowLeft") setPage((p) => Math.max(1, p - 1))
      if (event.key === "ArrowRight") setPage((p) => Math.min(doc?.pages ?? 1, p + 1))
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [doc])

  // -- derived ------------------------------------------------------------

  const resultKeys = Object.entries(results).map(([key, r]) => ({
    key,
    label: r.parser_name || r.parser_id,
    status: r.status,
  }))

  const resultCounts = React.useMemo(() => {
    const counts: Record<string, number> = {}
    Object.values(results).forEach((r) => {
      counts[r.parser_id] = (counts[r.parser_id] ?? 0) + 1
    })
    return counts
  }, [results])

  return (
    <SidebarProvider>
      <ParserSidebar
        parsers={parsers}
        loading={parsersLoading}
        selected={selected}
        onToggle={(id, on) =>
          setSelected((prev) => {
            const next = new Set(prev)
            if (on) next.add(id)
            else next.delete(id)
            return next
          })
        }
        configuring={configuring}
        onConfigure={setConfiguring}
        optionValues={optionValues}
        onOptionChange={(parserId, name, value) =>
          setOptionValues((prev) => ({ ...prev, [parserId]: { ...prev[parserId], [name]: value } }))
        }
        presets={presets}
        onSavePreset={async (parserId, name) => {
          try {
            // Store the effective options, defaults included, so a preset means
            // the same thing later even if a default moves.
            const spec = parsers.find((p) => p.id === parserId)
            const merged = Object.fromEntries(
              (spec?.options ?? []).map((o) => [o.name, optionValues[parserId]?.[o.name] ?? o.default])
            )
            const saved = await api.savePreset(name, parserId, merged)
            setPresets((prev) => [...prev.filter((p) => p.preset_id !== saved.preset_id), saved])
            toast.success(`Saved "${saved.name}"`)
          } catch (err) {
            toast.error("Could not save preset", { description: (err as Error).message })
          }
        }}
        onApplyPreset={(preset) => {
          setOptionValues((prev) => ({ ...prev, [preset.parser_id]: { ...preset.options } }))
          toast.success(`Applied "${preset.name}"`)
        }}
        onDeletePreset={async (presetId) => {
          await api.deletePreset(presetId)
          setPresets((prev) => prev.filter((p) => p.preset_id !== presetId))
        }}
        force={force}
        onForceChange={setForce}
        running={running}
        onRun={runSelected}
        resultCounts={resultCounts}
      />

      <SidebarInset className="h-svh min-w-0 overflow-hidden">
        <header className="bg-card flex h-12 shrink-0 items-center gap-2 border-b px-3">
          <SidebarTrigger />
          <Separator orientation="vertical" className="mr-1 h-4" />

          <Select value={doc?.doc_id ?? ""} onValueChange={(value) => void selectDocument(value)}>
            <SelectTrigger size="sm" className="h-8 w-[280px] text-xs">
              <SelectValue placeholder="No documents — add a PDF" />
            </SelectTrigger>
            <SelectContent>
              {documents.map((d) => (
                <SelectItem key={d.doc_id} value={d.doc_id} className="text-xs">
                  {d.name} · {d.pages}p
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <input
            ref={fileInput}
            type="file"
            accept="application/pdf"
            hidden
            onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])}
          />
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="outline" size="icon" className="size-8" onClick={() => fileInput.current?.click()}>
                <FileUp />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Add a PDF</TooltipContent>
          </Tooltip>

          {doc?.doc_class && (
            <Badge variant="secondary" className="hidden lg:inline-flex">
              {doc.doc_class}
              {doc.has_ground_truth ? " · ledger" : ""}
            </Badge>
          )}

          <Separator orientation="vertical" className="mx-1 h-4" />

          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="size-8"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft />
            </Button>
            <span className="text-muted-foreground w-14 text-center text-xs tabular-nums">
              {page} / {doc?.pages ?? "–"}
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="size-8"
              disabled={!doc || page >= doc.pages}
              onClick={() => setPage((p) => Math.min(doc?.pages ?? 1, p + 1))}
            >
              <ChevronRight />
            </Button>
          </div>

          <div className="ml-auto flex items-center gap-3">
            <div className="hidden items-center gap-2 md:flex">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={fit ? "secondary" : "ghost"}
                    size="icon"
                    className="size-8"
                    onClick={() => setFit((v) => !v)}
                  >
                    <Maximize2 />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Fit page width</TooltipContent>
              </Tooltip>
              <ZoomIn className="text-muted-foreground size-3.5" />
              <Slider
                className="w-24"
                min={0.5}
                max={3}
                step={0.1}
                disabled={fit}
                value={[zoom]}
                onValueChange={([v]) => {
                  setFit(false)
                  setZoom(v)
                }}
              />
            </div>
            <div className="flex items-center gap-1.5">
              <Tags className="text-muted-foreground size-3.5" />
              <Switch id="labels" checked={showLabels} onCheckedChange={setShowLabels} />
              <Label htmlFor="labels" className="text-muted-foreground text-xs font-normal">
                labels
              </Label>
            </div>
            <div className="flex items-center gap-1.5">
              <Columns2 className="text-muted-foreground size-3.5" />
              <Switch id="compare" checked={compare} onCheckedChange={setCompare} />
              <Label htmlFor="compare" className="text-muted-foreground text-xs font-normal">
                compare
              </Label>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="size-8"
              onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            >
              {resolvedTheme === "dark" ? <Sun /> : <Moon />}
            </Button>
          </div>
        </header>

        <div className="min-h-0 flex-1">
          {!doc ? (
            <div className="text-muted-foreground flex h-full items-center justify-center text-sm">
              No documents yet — add a PDF, or run <code className="mx-1">pdfplay samples</code> first.
            </div>
          ) : (
            <ResizablePanelGroup direction="horizontal">
              <ResizablePanel defaultSize={62} minSize={30}>
                <ResizablePanelGroup direction="horizontal">
                  <ResizablePanel defaultSize={compare ? 50 : 100} minSize={25}>
                    <PageViewer
                      docId={doc.doc_id}
                      page={page}
                      zoom={zoom}
                      fit={fit}
                      showLabels={showLabels}
                      result={leftKey ? results[leftKey] : undefined}
                      resultKeys={resultKeys}
                      activeKey={leftKey}
                      onKeyChange={setLeftKey}
                      layers={leftLayers}
                      onLayersChange={setLeftLayers}
                      onSelectBlock={setSelectedBlock}
                      selectedBlockId={selectedBlock?.id ?? null}
                    />
                  </ResizablePanel>
                  {compare && (
                    <>
                      <ResizableHandle withHandle />
                      <ResizablePanel defaultSize={50} minSize={25}>
                        <PageViewer
                          docId={doc.doc_id}
                          page={page}
                          zoom={zoom}
                          fit={fit}
                          showLabels={showLabels}
                          result={rightKey ? results[rightKey] : undefined}
                          resultKeys={resultKeys}
                          activeKey={rightKey}
                          onKeyChange={setRightKey}
                          layers={rightLayers}
                          onLayersChange={setRightLayers}
                          onSelectBlock={setSelectedBlock}
                          selectedBlockId={selectedBlock?.id ?? null}
                        />
                      </ResizablePanel>
                    </>
                  )}
                </ResizablePanelGroup>
              </ResizablePanel>

              <ResizableHandle withHandle />

              <ResizablePanel defaultSize={38} minSize={22}>
                <Inspector
                  docId={doc.doc_id}
                  page={page}
                  scores={scores}
                  result={leftKey ? results[leftKey] : undefined}
                  compareResult={rightKey ? results[rightKey] : undefined}
                  leftKey={leftKey}
                  rightKey={rightKey}
                  selectedBlock={selectedBlock}
                />
              </ResizablePanel>
            </ResizablePanelGroup>
          )}
        </div>
      </SidebarInset>
      <Toaster position="bottom-right" />
    </SidebarProvider>
  )
}
