import type { Metadata } from "next";
import { JetBrains_Mono, Orbitron } from "next/font/google";

import "highlight.js/styles/atom-one-dark.css";
import "./globals.css";

const display = Orbitron({ subsets: ["latin"], variable: "--font-display" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "STAR SARA — AI Executive Assistant",
  description:
    "Smart AI Response Assistant by STAR Technologies. Realtime voice, streaming reasoning, executive memory.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}
