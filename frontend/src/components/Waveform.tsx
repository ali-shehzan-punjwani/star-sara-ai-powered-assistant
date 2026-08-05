"use client";

interface Props {
  levels: number[];
  label: string;
  accent?: "royal" | "glow";
  active?: boolean;
}

/** Mirrored bar waveform used for both mic input and assistant output. */
export function Waveform({ levels, label, accent = "glow", active = true }: Props) {
  const color = accent === "royal" ? "#0B5ED7" : "#45C8FF";
  const peak = Math.max(...levels, 0.01);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        <span className="font-mono text-[10px] text-slate-500">
          {Math.round(peak * 100).toString().padStart(3, "0")}%
        </span>
      </div>
      <div className="flex h-14 items-center gap-[3px]">
        {levels.map((level, index) => {
          const height = Math.max(3, Math.min(1, level * 1.8) * 100);
          return (
            <span
              key={index}
              className="flex-1 rounded-full transition-[height,opacity] duration-100"
              style={{
                height: `${height}%`,
                background: `linear-gradient(180deg, ${color}, rgba(11,94,215,0.25))`,
                opacity: active ? 0.35 + (index / levels.length) * 0.65 : 0.15,
                boxShadow: level > 0.25 ? `0 0 10px ${color}66` : undefined,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}
