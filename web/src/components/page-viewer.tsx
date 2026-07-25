import * as React from "react"

import { LAYER_COLOR, LAYER_ORDER, type Block, type ParseResult } from "@/lib/api"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

interface Props {
  docId: string
  page: number
  zoom: number
  fit: boolean
  showLabels: boolean
  result: ParseResult | undefined
  resultKeys: { key: string; label: string; status: string }[]
  activeKey: string | null
  onKeyChange: (key: string) => void
  layers: string[]
  onLayersChange: (layers: string[]) => void
  onSelectBlock: (block: Block) => void
  selectedBlockId: string | null
}

/**
 * A rendered page with each parser's boxes drawn on top.
 *
 * Boxes are stored in PDF points with a top-left origin, so `zoom` (px per
 * point) is the only transform needed between the model and the screen.
 */
export function PageViewer({
  docId,
  page,
  zoom,
  fit,
  showLabels,
  result,
  resultKeys,
  activeKey,
  onKeyChange,
  layers,
  onLayersChange,
  onSelectBlock,
  selectedBlockId,
}: Props) {
  const pageResult = result?.pages.find((p) => p.page_number === page)
  const geometry = pageResult ?? { width: 612, height: 792, blocks: [] as Block[] }

  const availableLayers = React.useMemo(() => {
    const found = new Set<string>()
    result?.pages.forEach((p) => p.blocks.forEach((b) => found.add(b.layer)))
    return LAYER_ORDER.filter((l) => found.has(l))
  }, [result])

  // In fit mode each pane derives its own scale from its measured width, so
  // the two sides of a comparison stay legible at different panel sizes.
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const [fitZoom, setFitZoom] = React.useState(zoom)
  React.useEffect(() => {
    const node = scrollRef.current
    if (!node) return
    const measure = () => setFitZoom(Math.max(0.2, (node.clientWidth - 40) / geometry.width))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [geometry.width])

  const scale = fit ? fitZoom : zoom
  const boxes = (pageResult?.blocks ?? []).filter((b) => b.bbox && layers.includes(b.layer))

  return (
    <div className="flex h-full min-w-0 flex-col">
      <div className="bg-card flex h-11 shrink-0 items-center gap-2 border-b px-3">
        <Select value={activeKey ?? ""} onValueChange={onKeyChange}>
          <SelectTrigger size="sm" className="h-7 w-[190px] text-xs">
            <SelectValue placeholder="No results yet" />
          </SelectTrigger>
          <SelectContent>
            {resultKeys.map((r) => (
              <SelectItem key={r.key} value={r.key} className="text-xs">
                {r.label}
                {r.status === "error" ? " ⚠" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {availableLayers.length > 0 && (
          <ToggleGroup
            type="multiple"
            size="sm"
            variant="outline"
            value={layers}
            onValueChange={(value) => value.length && onLayersChange(value)}
          >
            {availableLayers.map((layer) => (
              <ToggleGroupItem key={layer} value={layer} className="h-7 px-2 text-[11px]">
                <span
                  className="mr-1 size-2 rounded-full"
                  style={{ backgroundColor: LAYER_COLOR[layer] }}
                  aria-hidden
                />
                {layer}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        )}

        {result?.status === "error" && (
          <Badge variant="destructive" className="ml-auto max-w-[240px] truncate">
            {result.error}
          </Badge>
        )}
        {result?.status === "ok" && (
          <Badge variant="secondary" className="ml-auto tabular-nums">
            {boxes.length} boxes · {result.duration_s.toFixed(2)}s
          </Badge>
        )}
      </div>

      <div ref={scrollRef} className="bg-muted/40 flex-1 overflow-auto p-4">
        <div
          className="relative mx-auto bg-white shadow-lg ring-1 ring-black/10"
          style={{ width: geometry.width * scale, height: geometry.height * scale }}
        >
          <img
            src={`/api/documents/${docId}/pages/${page}/image?scale=2`}
            alt={`Page ${page}`}
            className="block h-full w-full select-none"
            draggable={false}
          />
          {boxes.map((block) => (
            <Tooltip key={block.id}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={() => onSelectBlock(block)}
                  className={cn(
                    "absolute border transition-colors hover:bg-sky-400/25 focus-visible:outline-none",
                    selectedBlockId === block.id && "bg-sky-400/30 ring-2 ring-sky-500"
                  )}
                  style={{
                    left: block.bbox!.x0 * scale,
                    top: block.bbox!.y0 * scale,
                    width: Math.max(1, (block.bbox!.x1 - block.bbox!.x0) * scale),
                    height: Math.max(1, (block.bbox!.y1 - block.bbox!.y0) * scale),
                    borderColor: LAYER_COLOR[block.layer] ?? "#888",
                  }}
                >
                  {showLabels && block.kind !== "text" && (
                    <span
                      className="absolute -top-[13px] left-0 rounded-t px-1 text-[9px] leading-[13px] font-semibold text-white"
                      style={{ backgroundColor: LAYER_COLOR[block.layer] ?? "#888" }}
                    >
                      {block.kind}
                    </span>
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-md">
                <span className="text-muted-foreground/80 mr-1 font-mono text-[10px] uppercase">
                  {block.layer}/{block.kind}
                  {block.confidence != null ? ` ${(block.confidence * 100).toFixed(0)}%` : ""}
                </span>
                <span className="font-mono">{block.text.slice(0, 300) || "(no text)"}</span>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      </div>
    </div>
  )
}
