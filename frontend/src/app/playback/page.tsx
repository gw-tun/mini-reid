import { Suspense } from "react";

import PlaybackClient from "@/components/PlaybackClient";

export default function PlaybackPage() {
  return (
    <Suspense
      fallback={
        <p className="text-sm text-muted-foreground">Loading playback…</p>
      }
    >
      <div className="flex h-full min-h-0 flex-col">
        <PlaybackClient />
      </div>
    </Suspense>
  );
}
