"use client";

import { useEffect, useRef } from "react";

import type { AssistantState } from "@/lib/types";

const ROYAL = "11, 94, 215";
const GLOW = "69, 200, 255";

interface Particle {
  angle: number;
  radius: number;
  speed: number;
  size: number;
  drift: number;
}

interface Props {
  state: AssistantState;
  /** Rolling mic/output amplitude history, newest last, values 0..1. */
  levels: number[];
}

function makeParticles(count: number): Particle[] {
  return Array.from({ length: count }, () => ({
    angle: Math.random() * Math.PI * 2,
    radius: 0.55 + Math.random() * 0.75,
    speed: 0.0015 + Math.random() * 0.004,
    size: 0.6 + Math.random() * 1.9,
    drift: Math.random() * Math.PI * 2,
  }));
}

/**
 * The holographic core. One canvas, four behaviours:
 *  idle       - slow rotating energy sphere with orbiting particles
 *  listening  - concentric pulses driven by live mic amplitude
 *  thinking   - neural mesh of nodes wiring themselves together
 *  responding - radial voice wave driven by TTS output amplitude
 */
export function AICore({ state, levels }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(state);
  const levelsRef = useRef(levels);

  stateRef.current = state;
  levelsRef.current = levels;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    const particles = makeParticles(120);
    const nodes = Array.from({ length: 26 }, () => ({
      x: Math.random() * 2 - 1,
      y: Math.random() * 2 - 1,
      vx: (Math.random() - 0.5) * 0.0016,
      vy: (Math.random() - 0.5) * 0.0016,
    }));

    let frame = 0;
    let raf = 0;

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      const size = canvas.clientWidth;
      canvas.width = size * ratio;
      canvas.height = size * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const render = () => {
      frame += 1;
      const size = canvas.clientWidth;
      const cx = size / 2;
      const cy = size / 2;
      const unit = size / 2.6;
      const current = stateRef.current;
      const history = levelsRef.current;
      const level = history.length ? history[history.length - 1] : 0;
      const energy = current === "idle" ? 0.12 : Math.min(1, 0.25 + level * 1.6);

      context.clearRect(0, 0, size, size);

      // Core halo
      const halo = context.createRadialGradient(cx, cy, unit * 0.1, cx, cy, unit * 1.35);
      halo.addColorStop(0, `rgba(${GLOW}, ${0.34 + energy * 0.3})`);
      halo.addColorStop(0.45, `rgba(${ROYAL}, ${0.2 + energy * 0.18})`);
      halo.addColorStop(1, "rgba(0, 31, 84, 0)");
      context.fillStyle = halo;
      context.beginPath();
      context.arc(cx, cy, unit * 1.35, 0, Math.PI * 2);
      context.fill();

      // Rotating latitude/longitude sphere
      const spin = frame * (current === "thinking" ? 0.012 : 0.004);
      context.strokeStyle = `rgba(${GLOW}, 0.28)`;
      context.lineWidth = 1;
      for (let i = 0; i < 7; i += 1) {
        const t = (i / 6) * Math.PI;
        context.beginPath();
        context.ellipse(cx, cy, unit * Math.abs(Math.sin(t + spin)), unit, 0, 0, Math.PI * 2);
        context.stroke();
      }
      context.strokeStyle = `rgba(${ROYAL}, 0.35)`;
      for (let i = 1; i < 5; i += 1) {
        const ry = unit * Math.cos((i / 5) * Math.PI - Math.PI / 2);
        context.beginPath();
        context.ellipse(cx, cy, unit, Math.abs(ry), 0, 0, Math.PI * 2);
        context.stroke();
      }

      // Inner solid core
      const core = context.createRadialGradient(cx, cy, 0, cx, cy, unit * (0.34 + energy * 0.12));
      core.addColorStop(0, "rgba(255, 255, 255, 0.95)");
      core.addColorStop(0.35, `rgba(${GLOW}, 0.75)`);
      core.addColorStop(1, `rgba(${ROYAL}, 0)`);
      context.fillStyle = core;
      context.beginPath();
      context.arc(cx, cy, unit * (0.34 + energy * 0.12), 0, Math.PI * 2);
      context.fill();

      if (current === "thinking") {
        // Neural mesh: nodes drift and link when close enough.
        nodes.forEach((node) => {
          node.x += node.vx;
          node.y += node.vy;
          if (Math.abs(node.x) > 1) node.vx *= -1;
          if (Math.abs(node.y) > 1) node.vy *= -1;
        });
        for (let i = 0; i < nodes.length; i += 1) {
          for (let j = i + 1; j < nodes.length; j += 1) {
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const distance = Math.hypot(dx, dy);
            if (distance > 0.55) continue;
            context.strokeStyle = `rgba(${GLOW}, ${0.32 * (1 - distance / 0.55)})`;
            context.beginPath();
            context.moveTo(cx + nodes[i].x * unit, cy + nodes[i].y * unit);
            context.lineTo(cx + nodes[j].x * unit, cy + nodes[j].y * unit);
            context.stroke();
          }
        }
        nodes.forEach((node) => {
          context.fillStyle = `rgba(255,255,255,0.8)`;
          context.beginPath();
          context.arc(cx + node.x * unit, cy + node.y * unit, 1.8, 0, Math.PI * 2);
          context.fill();
        });
      } else {
        // Orbiting particle field
        particles.forEach((particle) => {
          particle.angle += particle.speed * (1 + energy * 3);
          const wobble = Math.sin(frame * 0.02 + particle.drift) * 0.05;
          const radius = unit * (particle.radius + wobble + energy * 0.12);
          const x = cx + Math.cos(particle.angle) * radius;
          const y = cy + Math.sin(particle.angle) * radius * 0.62;
          context.fillStyle = `rgba(${GLOW}, ${0.25 + energy * 0.5})`;
          context.beginPath();
          context.arc(x, y, particle.size, 0, Math.PI * 2);
          context.fill();
        });
      }

      if (current === "listening") {
        // Amplitude rings expanding outward from the core.
        for (let i = 0; i < 3; i += 1) {
          const phase = ((frame * 0.012 + i / 3) % 1);
          context.strokeStyle = `rgba(${GLOW}, ${(1 - phase) * (0.25 + level)})`;
          context.lineWidth = 2;
          context.beginPath();
          context.arc(cx, cy, unit * (0.4 + phase * 1.05), 0, Math.PI * 2);
          context.stroke();
        }
      }

      if (current === "responding") {
        // Radial voice wave: history mapped around the circle.
        const points = history.length || 1;
        context.strokeStyle = `rgba(255,255,255,0.75)`;
        context.lineWidth = 2;
        context.beginPath();
        for (let i = 0; i <= points; i += 1) {
          const amplitude = history[i % points] ?? 0;
          const angle = (i / points) * Math.PI * 2 - Math.PI / 2;
          const radius = unit * (0.72 + amplitude * 0.6);
          const x = cx + Math.cos(angle) * radius;
          const y = cy + Math.sin(angle) * radius;
          if (i === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        context.closePath();
        context.stroke();
      }

      raf = requestAnimationFrame(render);
    };

    raf = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="aspect-square w-full" aria-hidden />;
}
