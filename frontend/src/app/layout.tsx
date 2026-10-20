import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "mini-reid",
  description: "Person re-identification across cameras (VIP-9), to merge into semantic-va",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <header className="border-b border-zinc-200 dark:border-zinc-800">
          <nav className="mx-auto flex w-full max-w-6xl items-center gap-6 px-4 py-3 text-sm">
            <Link href="/" className="font-semibold">
              mini-reid
            </Link>
            <Link href="/" className="text-zinc-600 hover:underline dark:text-zinc-400">
              Recordings
            </Link>
            <Link href="/tracks" className="text-zinc-600 hover:underline dark:text-zinc-400">
              Person tracks
            </Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
