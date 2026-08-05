"use client";

import { Mic, MicOff, Square } from "lucide-react";

import { AICore } from "@/components/AICore";
import { Waveform } from "@/components/Waveform";
import type { AccuracyMode, AssistantState } from "@/lib/types";

const STATE_COPY: Record<AssistantState, { label: string; hint: string }> = {
  idle: { label: "STANDING BY", hint: "Awaiting wake word" },
  listening: { label: "LISTENING", hint: "Capturing your voice" },
  thinking: { label: "THINKING", hint: "Reasoning over context" },
  responding: { label: "RESPONDING", hint: "Speaking" },
};

const MODES: AccuracyMode[] = ["fast", "balanced", "accurate"];

interface Props {
  state: AssistantState;
  micActive: boolean;
  inputLevels: number[];
  outputLevels: number[];
  mode: AccuracyMode;
  alwaysOn: boolean;
  onToggleMic: () => void;
  onInterrupt: () => void;
  onModeChange: (mode: AccuracyMode) => void;
  onAlwaysOnChange: (enabled: boolean) => void;
}

export function CoreConsole({
  state,
  micActive,
  inputLevels,
  outputLevels,
  mode,
  alwaysOn,
  onToggleMic,
  onInterrupt,
  onModeChange,
  onAlwaysOnChange,
}: Props) {
  const copy = STATE_COPY[state];
  const coreLevels = state === "responding" ? outputLevels : inputLevels;

  return (
    <div className="panel panel-glow flex h-full flex-col gap-4 p-5">
      <div className="relative mx-auto w-full max-w-[320px]">
        <AICore state={state} levels={coreLevels} />
        <div className="pointer-events-none absolute inset-x-0 bottom-1 text-center">
          <p className="font-display text-sm tracking-[0.4em] text-white">{copy.label}</p>
          <p className="text-[11px] text-slate-400">{copy.hint}</p>
        </div>
      </div>

      <Waveform levels={inputLevels} label="User voice input" accent="glow" active={micActive} />
      <Waveform
        levels={outputLevels}
        label="Assistant voice output"
        accent="royal"
        active={state === "responding"}
      />

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggleMic}
          className={`flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition ${
            micActive
              ? "bg-glow text-navy hover:bg-glow/80"
              : "bg-royal text-white hover:bg-royal/80"
          }`}
        >
          {micActive ? <Mic size={16} /> : <MicOff size={16} />}
          {micActive ? "Live" : "Activate microphone"}
        </button>
        <button
          type="button"
          onClick={onInterrupt}
          className="rounded-xl border border-white/10 p-3 text-slate-300 transition hover:border-rose-400/60 hover:text-rose-300"
          aria-label="Interrupt"
        >
          <Square size={16} />
        </button>
      </div>

      <div className="space-y-3 border-t border-white/10 pt-3">
        <div>
          <p className="label mb-2">Recognition mode</p>
          <div className="grid grid-cols-3 gap-1 rounded-xl bg-black/30 p-1">
            {MODES.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => onModeChange(option)}
                className={`rounded-lg py-1.5 text-[11px] font-medium capitalize transition ${
                  mode === option ? "bg-royal text-white" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
        <label className="flex cursor-pointer items-center justify-between text-[11px] text-slate-400">
          <span>Continuous conversation (no wake word)</span>
          <input
            type="checkbox"
            checked={alwaysOn}
            onChange={(event) => onAlwaysOnChange(event.target.checked)}
            className="h-4 w-4 accent-[#45C8FF]"
          />
        </label>
      </div>
    </div>
  );
}
