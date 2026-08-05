"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { StreamingAudioPlayer } from "@/lib/audioPlayer";
import { WS_URL } from "@/lib/api";
import type { AccuracyMode, AssistantState, ChatMessage, ServerEvent, TurnMetrics } from "@/lib/types";

const LEVEL_HISTORY = 64;

export interface VoiceSession {
  connected: boolean;
  micActive: boolean;
  state: AssistantState;
  messages: ChatMessage[];
  inputLevels: number[];
  outputLevels: number[];
  metrics: TurnMetrics | null;
  error: string | null;
  startMic: () => Promise<void>;
  stopMic: () => void;
  sendText: (text: string) => void;
  interrupt: () => void;
  setAccuracyMode: (mode: AccuracyMode) => void;
  setAlwaysOn: (enabled: boolean) => void;
}

function id() {
  return Math.random().toString(36).slice(2, 10);
}

export function useVoiceSession(): VoiceSession {
  const socketRef = useRef<WebSocket | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const playerRef = useRef<StreamingAudioPlayer | null>(null);
  const currentTurnRef = useRef<string | null>(null);

  const [connected, setConnected] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [state, setState] = useState<AssistantState>("idle");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputLevels, setInputLevels] = useState<number[]>(() => new Array(LEVEL_HISTORY).fill(0));
  const [outputLevels, setOutputLevels] = useState<number[]>(() => new Array(LEVEL_HISTORY).fill(0));
  const [metrics, setMetrics] = useState<TurnMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pushLevel = useCallback(
    (setter: typeof setInputLevels, value: number) =>
      setter((previous) => [...previous.slice(1), value]),
    [],
  );

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case "state":
        if (event.state) setState(event.state);
        break;
      case "wake":
        setState("listening");
        break;
      case "transcript":
        if (!event.text) break;
        setMessages((previous) => [
          ...previous,
          { id: id(), role: "user", content: event.text!, timestamp: Date.now(), voice: true },
        ]);
        break;
      case "token": {
        if (!event.text) break;
        const turn = event.turn_id ?? "turn";
        setMessages((previous) => {
          const last = previous[previous.length - 1];
          if (last?.role === "assistant" && last.id === turn) {
            return [...previous.slice(0, -1), { ...last, content: last.content + event.text! }];
          }
          return [
            ...previous,
            {
              id: turn,
              role: "assistant",
              content: event.text!,
              timestamp: Date.now(),
              streaming: true,
            },
          ];
        });
        currentTurnRef.current = turn;
        break;
      }
      case "reply_done":
        setMessages((previous) =>
          previous.map((message) =>
            message.id === event.turn_id ? { ...message, streaming: false } : message,
          ),
        );
        break;
      case "audio":
        if (event.audio) playerRef.current?.enqueue(event.audio);
        break;
      case "metrics": {
        const next = (event.data ?? {}) as unknown as TurnMetrics;
        setMetrics(next);
        setMessages((previous) =>
          previous.map((message) =>
            message.id === event.turn_id ? { ...message, metrics: next } : message,
          ),
        );
        break;
      }
      case "error":
        setError(event.text ?? "Unknown error");
        break;
      default:
        break;
    }
  }, []);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState <= WebSocket.OPEN) return socketRef.current;
    const socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      setConnected(true);
      setError(null);
    };
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setError("Cannot reach the STAR SARA backend.");
    socket.onmessage = (message) => handleEvent(JSON.parse(message.data) as ServerEvent);
    socketRef.current = socket;
    return socket;
  }, [handleEvent]);

  useEffect(() => {
    connect();
    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connect]);

  // Output level meter driven by the TTS analyser.
  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const analyser = playerRef.current?.outputAnalyser;
      if (analyser) {
        const data = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (const sample of data) peak = Math.max(peak, Math.abs(sample - 128) / 128);
        pushLevel(setOutputLevels, peak);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [pushLevel]);

  const startMic = useCallback(async () => {
    try {
      const socket = connect();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const context = new AudioContext();
      await context.audioWorklet.addModule("/worklets/pcm-recorder.js");
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, "pcm-recorder", {
        processorOptions: { targetSampleRate: 16000 },
      });
      node.port.onmessage = ({ data }) => {
        pushLevel(setInputLevels, data.peak as number);
        if (socket.readyState === WebSocket.OPEN) socket.send(data.pcm as ArrayBuffer);
      };
      source.connect(node);

      contextRef.current = context;
      streamRef.current = stream;
      nodeRef.current = node;
      playerRef.current = new StreamingAudioPlayer(context);
      setMicActive(true);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Microphone unavailable");
    }
  }, [connect, pushLevel]);

  const stopMic = useCallback(() => {
    nodeRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    void contextRef.current?.close();
    nodeRef.current = null;
    streamRef.current = null;
    contextRef.current = null;
    playerRef.current = null;
    setMicActive(false);
    setInputLevels(new Array(LEVEL_HISTORY).fill(0));
  }, []);

  useEffect(() => stopMic, [stopMic]);

  const send = useCallback((payload: object) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  }, []);

  const sendText = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      if (!playerRef.current && contextRef.current) {
        playerRef.current = new StreamingAudioPlayer(contextRef.current);
      }
      setMessages((previous) => [
        ...previous,
        { id: id(), role: "user", content: text, timestamp: Date.now() },
      ]);
      send({ type: "text", text, data: { speak: micActive } });
    },
    [micActive, send],
  );

  const interrupt = useCallback(() => {
    playerRef.current?.stop();
    send({ type: "interrupt" });
    setState("idle");
  }, [send]);

  const setAccuracyMode = useCallback(
    (mode: AccuracyMode) => send({ type: "config", data: { accuracy_mode: mode } }),
    [send],
  );

  const setAlwaysOn = useCallback(
    (enabled: boolean) => send({ type: "config", data: { always_on: enabled } }),
    [send],
  );

  return {
    connected,
    micActive,
    state,
    messages,
    inputLevels,
    outputLevels,
    metrics,
    error,
    startMic,
    stopMic,
    sendText,
    interrupt,
    setAccuracyMode,
    setAlwaysOn,
  };
}
