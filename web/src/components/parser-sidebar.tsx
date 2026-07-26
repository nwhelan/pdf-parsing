import { CircleAlert, CircleDot, Cloud, Cpu, Play, Settings2 } from "lucide-react"

import type { ParserSpec } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  SidebarRail,
  SidebarSeparator,
} from "@/components/ui/sidebar"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

type OptionValues = Record<string, Record<string, unknown>>

interface Props {
  parsers: ParserSpec[]
  loading: boolean
  selected: Set<string>
  onToggle: (id: string, on: boolean) => void
  configuring: string | null
  onConfigure: (id: string) => void
  optionValues: OptionValues
  onOptionChange: (parserId: string, name: string, value: unknown) => void
  force: boolean
  onForceChange: (value: boolean) => void
  running: boolean
  onRun: () => void
  resultCounts: Record<string, number>
}

export function ParserSidebar({
  parsers,
  loading,
  selected,
  onToggle,
  configuring,
  onConfigure,
  optionValues,
  onOptionChange,
  force,
  onForceChange,
  running,
  onRun,
  resultCounts,
}: Props) {
  const current = parsers.find((p) => p.id === configuring) ?? null
  const local = parsers.filter((p) => p.kind === "local")
  const remote = parsers.filter((p) => p.kind === "remote")

  const renderGroup = (label: string, icon: React.ReactNode, items: ParserSpec[]) => (
    <SidebarGroup>
      <SidebarGroupLabel className="gap-1.5">
        {icon}
        {label}
      </SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {items.map((parser) => (
            <SidebarMenuItem key={parser.id}>
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`parser-${parser.id}`}
                  checked={selected.has(parser.id)}
                  disabled={!parser.available}
                  onCheckedChange={(value) => onToggle(parser.id, value === true)}
                  aria-label={`Run ${parser.name}`}
                />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <SidebarMenuButton
                      isActive={configuring === parser.id}
                      onClick={() => onConfigure(parser.id)}
                      className={parser.available ? "" : "text-muted-foreground"}
                    >
                      <span className="truncate">{parser.name}</span>
                      {!parser.available && <CircleAlert className="ml-auto size-3.5 opacity-60" />}
                    </SidebarMenuButton>
                  </TooltipTrigger>
                  <TooltipContent side="right" className="max-w-xs">
                    {parser.available ? parser.description : parser.unavailable_reason}
                  </TooltipContent>
                </Tooltip>
              </div>
              {resultCounts[parser.id] ? (
                <SidebarMenuBadge
                  // Normally positioned by a peer selector on the menu button; the
                  // checkbox wrapper below breaks that sibling chain, so set it here.
                  className="top-1.5 right-2"
                >{resultCounts[parser.id]}</SidebarMenuBadge>
              ) : null}
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  )

  return (
    <Sidebar collapsible="offcanvas">
      <SidebarHeader>
        <div className="flex items-center gap-2 px-2 py-1">
          <CircleDot className="size-4 text-primary" />
          <span className="text-sm font-semibold tracking-tight">pdfplay</span>
          <Badge variant="secondary" className="ml-auto text-[10px]">
            {parsers.filter((p) => p.available).length}/{parsers.length} ready
          </Badge>
        </div>
      </SidebarHeader>

      <SidebarContent>
        {loading ? (
          <SidebarGroup>
            <SidebarGroupContent>
              <SidebarMenu>
                {Array.from({ length: 6 }).map((_, i) => (
                  <SidebarMenuItem key={i}>
                    <SidebarMenuSkeleton showIcon />
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ) : (
          <>
            {renderGroup("Local", <Cpu className="size-3.5" />, local)}
            {renderGroup("Remote", <Cloud className="size-3.5" />, remote)}
            <SidebarSeparator />
            <SidebarGroup>
              <SidebarGroupLabel className="gap-1.5">
                <Settings2 className="size-3.5" />
                {current ? current.name : "Options"}
              </SidebarGroupLabel>
              <SidebarGroupContent className="px-2">
                {!current ? (
                  <p className="text-muted-foreground text-xs">Pick a parser to configure it.</p>
                ) : (
                  <div className="space-y-3">
                    <p className="text-muted-foreground text-xs leading-relaxed">{current.description}</p>
                    {current.kind === "remote" && (
                      <p className="text-muted-foreground text-xs">
                        <span className="font-medium">Cost:</span> {current.cost_hint}
                      </p>
                    )}
                    {current.options.length === 0 && (
                      <p className="text-muted-foreground text-xs">No options.</p>
                    )}
                    {current.options.map((opt) => {
                      const value = optionValues[current.id]?.[opt.name] ?? opt.default
                      return (
                        <div key={opt.name} className="space-y-1">
                          <div className="flex items-center justify-between gap-2">
                            <Label htmlFor={`${current.id}-${opt.name}`} className="text-xs font-normal">
                              {opt.label}
                            </Label>
                            {opt.type === "bool" ? (
                              <Switch
                                id={`${current.id}-${opt.name}`}
                                checked={Boolean(value)}
                                onCheckedChange={(v) => onOptionChange(current.id, opt.name, v)}
                              />
                            ) : opt.type === "choice" ? (
                              <Select
                                value={String(value)}
                                onValueChange={(v) =>
                                  onOptionChange(
                                    current.id,
                                    opt.name,
                                    typeof opt.default === "number" ? Number(v) : v
                                  )
                                }
                              >
                                <SelectTrigger size="sm" className="h-7 w-32 text-xs">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  {(opt.choices ?? []).map((choice) => (
                                    <SelectItem key={String(choice)} value={String(choice)} className="text-xs">
                                      {String(choice)}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            ) : (
                              <Input
                                id={`${current.id}-${opt.name}`}
                                className="h-7 w-24 text-xs"
                                type={opt.type === "str" ? "text" : "number"}
                                step={opt.type === "int" ? 1 : "any"}
                                value={String(value ?? "")}
                                onChange={(e) =>
                                  onOptionChange(
                                    current.id,
                                    opt.name,
                                    opt.type === "int"
                                      ? Number.parseInt(e.target.value, 10)
                                      : opt.type === "float"
                                        ? Number.parseFloat(e.target.value)
                                        : e.target.value
                                  )
                                }
                              />
                            )}
                          </div>
                          {opt.help && <p className="text-muted-foreground text-[11px] leading-snug">{opt.help}</p>}
                        </div>
                      )
                    })}
                  </div>
                )}
              </SidebarGroupContent>
            </SidebarGroup>
          </>
        )}
      </SidebarContent>

      <SidebarFooter>
        <div className="flex items-center justify-between px-2">
          <Label htmlFor="force-rerun" className="text-muted-foreground text-xs font-normal">
            Ignore cache
          </Label>
          <Switch id="force-rerun" checked={force} onCheckedChange={onForceChange} />
        </div>
        <Button onClick={onRun} disabled={running || selected.size === 0} className="w-full">
          <Play />
          {running ? "Running…" : `Run ${selected.size || ""}`.trim()}
        </Button>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
