import type { Metadata } from "next";

import LivePlayer from "@/components/LivePlayer";

export const metadata: Metadata = {
  title: "Live View",
};

export default function LivePage() {
  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="shrink-0">
        <h1 className="text-2xl font-semibold tracking-tight">Live View Monitoring</h1>
        <p className="text-sm text-muted-foreground">
          MediaMTX cam01 · WebRTC (WHEP) · looped recording feed with YOLO overlay
        </p>
      </div>
      <LivePlayer />
    </div>
  );
}
