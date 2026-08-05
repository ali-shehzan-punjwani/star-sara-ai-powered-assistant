"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Chat } from "@/components/Chat";
import { CoreConsole } from "@/components/CoreConsole";
import { Profile } from "@/components/Profile";
import {
  EngineCard,
  MemoryCard,
  StatusCard,
  SystemCard,
  TasksCard,
} from "@/components/StatCards";
import { useVoiceSession } from "@/hooks/useVoiceSession";
import { fetchStatus, fetchSystem, fetchTasks } from "@/lib/api";
import type { AccuracyMode, AssistantStatus, SystemStats, Task } from "@/lib/types";

export default function Dashboard() {
  const session = useVoiceSession();
  const [status, setStatus] = useState<AssistantStatus | null>(null);
  const [system, setSystem] = useState<SystemStats | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [mode, setMode] = useState<AccuracyMode>("fast");
  const [alwaysOn, setAlwaysOn] = useState(false);
  // /api/status reports the server default; the live mode is per-connection, so
  // only the first load may seed it — later polls must not undo a user choice.
  const modeSeeded = useRef(false);

  useEffect(() => {
    const load = async () => {
      try {
        const [nextStatus, nextTasks] = await Promise.all([fetchStatus(), fetchTasks()]);
        setStatus(nextStatus);
        if (!modeSeeded.current) {
          setMode(nextStatus.engine.accuracy_mode);
          modeSeeded.current = true;
        }
        setTasks(nextTasks.tasks.filter((task) => task.status === "pending"));
      } catch {
        setStatus(null);
      }
    };
    void load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [session.messages.length]);

  useEffect(() => {
    const poll = async () => {
      try {
        setSystem(await fetchSystem());
      } catch {
        setSystem(null);
      }
    };
    void poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  const toggleMic = useCallback(() => {
    if (session.micActive) session.stopMic();
    else void session.startMic();
  }, [session]);

  const changeMode = useCallback(
    (next: AccuracyMode) => {
      setMode(next);
      session.setAccuracyMode(next);
    },
    [session],
  );

  const changeAlwaysOn = useCallback(
    (next: boolean) => {
      setAlwaysOn(next);
      session.setAlwaysOn(next);
    },
    [session],
  );

  return (
    <main className="mx-auto flex min-h-screen max-w-[1500px] flex-col gap-4 p-4 lg:p-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-2xl tracking-[0.32em] text-white">STAR SARA</h1>
          <p className="text-[11px] uppercase tracking-[0.3em] text-slate-500">
            AI Executive Assistant Platform · STAR Technologies
          </p>
        </div>
        <div className="flex items-center gap-3">
          {session.error && (
            <p className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-[11px] text-rose-200">
              {session.error}
            </p>
          )}
          <Profile status={status} />
        </div>
      </header>

      <div className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-[minmax(320px,380px)_1fr_minmax(260px,300px)]">
        <CoreConsole
          state={session.state}
          micActive={session.micActive}
          inputLevels={session.inputLevels}
          outputLevels={session.outputLevels}
          mode={mode}
          alwaysOn={alwaysOn}
          onToggleMic={toggleMic}
          onInterrupt={session.interrupt}
          onModeChange={changeMode}
          onAlwaysOnChange={changeAlwaysOn}
        />

        <div className="min-h-[520px] lg:h-[calc(100vh-9rem)]">
          <Chat messages={session.messages} onSend={session.sendText} disabled={!session.connected} />
        </div>

        <div className="grid content-start gap-4">
          <StatusCard status={status} connected={session.connected} />
          <MemoryCard status={status} />
          <TasksCard tasks={tasks} />
          <SystemCard stats={system} />
          <EngineCard status={status} metrics={session.metrics} />
        </div>
      </div>
    </main>
  );
}
