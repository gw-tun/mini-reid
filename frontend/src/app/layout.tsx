import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { AppSidebar } from "@/components/app-sidebar";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "mini-reid",
    template: "%s · mini-reid",
  },
  description: "Person re-identification practice project for semantic-va",
};

const themeScript = `(function(){try{var s=localStorage.getItem("mini-reid-theme");var d=s?s==="dark":window.matchMedia("(prefers-color-scheme: dark)").matches;document.documentElement.classList.toggle("dark",d);}catch(e){}})();`;

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="h-full overflow-hidden bg-background text-foreground">
        <TooltipProvider>
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset className="min-h-0">
              <header className="flex h-14 shrink-0 items-center gap-3 border-b bg-background px-4">
                <SidebarTrigger className="-ml-1" />
                <div className="flex items-baseline gap-2">
                  <span className="text-sm font-semibold tracking-tight">mini-reid</span>
                  <span className="hidden text-xs text-muted-foreground sm:inline">
                    person re-identification console
                  </span>
                </div>
              </header>
              <main className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden p-4">
                {children}
              </main>
            </SidebarInset>
          </SidebarProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
