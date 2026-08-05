"use client";

import { Activity, Battery, Brain, Cpu, ListChecks, Signal, Wifi } from "lucide-react";
import type { ReactNode } from "react";

import type { AssistantStatus, SystemStats, Task, TurnMetrics } from "@/lib/types";

function Card({
  title,
  icon,
  children,
  accent,
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
  accent?: boolean;
}) {
  return (
    <section className={`panel ${accent ? "panel-glow" : ""} p-4`}>
      <header className="mb-3 flex items-center gap-2 text-slate-400">
        {icon}
        <h2 className="label">{title}</h2>
      </header>
      {children}
    </section>
  );
}

function Meter({ label, value, suffix = "%" }: { label: string; value: number | null; suffix?: string }) {
  const pct = value ?? 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[11px] text-slate-400">
        <span>{label}</span>
        <span className="font-mono text-slate-200">{value === null ? "—" : `${Math.round(pct)}${suffix}`}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-gradient-to-r from-royal to-glow transition-all duration-500"
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
    </div>
  );
}

export function StatusCard({ status, connected }: { status: AssistantStatus | null; connected: boolean }) {
  const online = connected && Boolean(status?.online);
  return (
    <Card title="Assistant Status" icon={<Activity size={14} />} accent>
      <div className="flex items-center gap-3">
        <span className="relative flex h-3 w-3">
          {online && (
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-glow opacity-70" />
          )}
          <span
            className={`relative inline-flex h-3 w-3 rounded-full ${online ? "bg-glow" : "bg-rose-500"}`}
          />
        </span>
        <p className="font-mono text-lg font-semibold tracking-wide text-white">
          {status?.assistant ?? "STAR SARA"} {online ? "ONLINE" : "OFFLINE"}
        </p>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Wake words: <span className="text-slate-200">&ldquo;STAR SARA&rdquo;, &ldquo;SARA&rdquo;</span> ·{" "}
        {status?.engine.wake_word ?? "—"}
      </p>
    </Card>
  );
}

export function MemoryCard({ status }: { status: AssistantStatus | null }) {
  return (
    <Card title="Memory" icon={<Brain size={14} />}>
      <p className="stat">{status?.counts.memories ?? 0}</p>
      <p className="text-xs text-slate-400">stored memories · {status?.counts.notes ?? 0} notes</p>
    </Card>
  );
}

export function TasksCard({ tasks }: { tasks: Task[] }) {
  return (
    <Card title="Tasks" icon={<ListChecks size={14} />}>
      <p className="stat">{tasks.length}</p>
      <ul className="mt-2 space-y-1 text-xs text-slate-300">
        {tasks.slice(0, 3).map((task) => (
          <li key={task.id} className="flex items-start gap-2">
            <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-glow" />
            <span className="line-clamp-1">{task.task}</span>
          </li>
        ))}
        {tasks.length === 0 && <li className="text-slate-500">No upcoming reminders.</li>}
      </ul>
    </Card>
  );
}

export function SystemCard({ stats }: { stats: SystemStats | null }) {
  return (
    <Card title="System" icon={<Cpu size={14} />}>
      <div className="space-y-2">
        <Meter label="CPU" value={stats?.cpu_percent ?? null} />
        <Meter label="RAM" value={stats?.ram_percent ?? null} />
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span className="flex items-center gap-1">
            <Battery size={12} />
            {stats?.battery ? `${stats.battery.percent}%` : "AC"}
          </span>
          <span className="flex items-center gap-1">
            <Wifi size={12} className={stats?.network ? "text-glow" : "text-rose-400"} />
            {stats?.network ? "Online" : "Offline"}
          </span>
        </div>
      </div>
    </Card>
  );
}

export function EngineCard({ status, metrics }: { status: AssistantStatus | null; metrics: TurnMetrics | null }) {
  const engine = status?.engine;
  const rows: Array<[string, string]> = [
    ["LLM", engine?.llm ?? "—"],
    ["STT", `${engine?.stt ?? "—"} · ${engine?.stt_device ?? "cpu"}`],
    ["TTS", engine?.tts_voice ?? "—"],
  ];
  const latency: Array<[string, number | null | undefined]> = [
    ["STT", metrics?.stt_ms],
    ["1st token", metrics?.first_token_ms],
    ["1st audio", metrics?.first_audio_ms],
  ];

  return (
    <Card title="AI Engine" icon={<Signal size={14} />}>
      <dl className="space-y-1 text-[11px]">
        {rows.map(([key, value]) => (
          <div key={key} className="flex justify-between gap-3">
            <dt className="text-slate-500">{key}</dt>
            <dd className="truncate font-mono text-slate-200">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 grid grid-cols-3 gap-2 border-t border-white/10 pt-2">
        {latency.map(([key, value]) => (
          <div key={key}>
            <p className="text-[9px] uppercase tracking-wider text-slate-500">{key}</p>
            <p className="font-mono text-sm text-glow">
              {value == null ? "—" : `${Math.round(value)}ms`}
            </p>
          </div>
        ))}
      </div>
    </Card>
  );
}
