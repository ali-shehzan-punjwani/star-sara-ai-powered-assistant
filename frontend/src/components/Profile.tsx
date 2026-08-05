"use client";

import type { AssistantStatus } from "@/lib/types";

export function Profile({ status }: { status: AssistantStatus | null }) {
  const owner = status?.owner;
  const initials = (owner?.name ?? "Ali Shehzan Punjwani")
    .split(" ")
    .map((part) => part[0])
    .slice(0, 2)
    .join("");

  return (
    <section className="panel flex items-center gap-3 p-4">
      <div className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-royal to-navy font-semibold text-white ring-1 ring-glow/40">
        {initials}
      </div>
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold text-white">
          {owner?.name ?? "Ali Shehzan Punjwani"}
        </p>
        <p className="truncate text-[11px] text-slate-400">
          {owner?.title ?? "Founder & CEO"} · {owner?.company ?? "STAR Technologies"}
        </p>
      </div>
    </section>
  );
}
