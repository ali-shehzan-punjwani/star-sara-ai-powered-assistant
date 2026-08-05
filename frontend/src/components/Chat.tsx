"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Mic, SendHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import type { ChatMessage } from "@/lib/types";

function timestamp(value: number) {
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function Bubble({ message }: { message: ChatMessage }) {
  const mine = message.role === "user";
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 320, damping: 28 }}
      className={`flex ${mine ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          mine
            ? "rounded-br-sm bg-gradient-to-br from-royal to-navy text-white"
            : "rounded-bl-sm border border-white/10 bg-white/[0.05] text-slate-100"
        }`}
      >
        <div className="prose prose-invert prose-sm max-w-none prose-p:my-1 prose-pre:my-2">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
            {message.content}
          </ReactMarkdown>
        </div>
        <div className="mt-1 flex items-center gap-2 text-[10px] text-white/50">
          {message.voice && <Mic size={10} />}
          <span>{timestamp(message.timestamp)}</span>
          {message.metrics?.first_audio_ms != null && (
            <span className="font-mono">voice {Math.round(message.metrics.first_audio_ms)}ms</span>
          )}
          {message.streaming && <span className="animate-pulse">▍</span>}
        </div>
      </div>
    </motion.div>
  );
}

interface Props {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function Chat({ messages, onSend, disabled }: Props) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    if (!draft.trim()) return;
    onSend(draft.trim());
    setDraft("");
  };

  return (
    <div className="panel panel-glow flex h-full min-h-0 flex-col">
      <header className="flex items-center justify-between border-b border-white/10 px-5 py-3">
        <h2 className="label">Conversation</h2>
        <span className="font-mono text-[10px] text-slate-500">{messages.length} messages</span>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
        {messages.length === 0 && (
          <p className="mt-10 text-center text-sm text-slate-500">
            Say <span className="text-glow">&ldquo;STAR SARA&rdquo;</span> or type below to begin.
          </p>
        )}
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <Bubble key={`${message.id}-${message.timestamp}`} message={message} />
          ))}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      <div className="flex items-center gap-2 border-t border-white/10 p-3">
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && submit()}
          placeholder="Ask STAR SARA anything…"
          disabled={disabled}
          className="flex-1 rounded-xl border border-white/10 bg-black/30 px-4 py-2.5 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-glow/60"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !draft.trim()}
          className="rounded-xl bg-royal p-2.5 text-white transition hover:bg-glow hover:text-navy disabled:opacity-40"
          aria-label="Send message"
        >
          <SendHorizontal size={16} />
        </button>
      </div>
    </div>
  );
}
