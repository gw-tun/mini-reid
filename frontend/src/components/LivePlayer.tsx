"use client";

import { CircleAlert, LoaderCircle, Radio, Scan, Video } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { VideoStage } from "@/components/video-stage";

type Status = "connecting" | "live" | "error";

type LiveBox = {
  label: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

const CAMERA = "cam01";
const CAMERA_ASPECT = 16 / 9;

type Box = { x1: number; y1: number; x2: number; y2: number };

function intersectionOverUnion(a: Box, b: Box): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (overlap <= 0) {
    return 0;
  }
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - overlap;
  return union > 0 ? overlap / union : 0;
}

/** Keep the strongest box per region so stacked detections do not clutter the frame. */
function suppressOverlaps(boxes: LiveBox[], threshold = 0.25): LiveBox[] {
  const sorted = [...boxes].sort((left, right) => right.score - left.score);
  const kept: LiveBox[] = [];
  for (const candidate of sorted) {
    const overlaps = kept.some(
      (existing) => intersectionOverUnion(existing, candidate) > threshold,
    );
    if (!overlaps) {
      kept.push(candidate);
    }
  }
  return kept;
}

export default function LivePlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const [fps, setFps] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detectOn, setDetectOn] = useState(false);
  const [boxes, setBoxes] = useState<LiveBox[]>([]);

  const stop = useCallback(() => {
    pcRef.current?.close();
    pcRef.current = null;
  }, []);

  const start = useCallback(async () => {
    stop();
    const pc = new RTCPeerConnection();
    pcRef.current = pc;
    const stream = new MediaStream();
    pc.ontrack = (event) => {
      stream.addTrack(event.track);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setStatus("live");
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed" || pc.connectionState === "disconnected") {
        setError(`peer connection ${pc.connectionState}`);
        setStatus("error");
      }
    };
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === "complete") {
          resolve();
          return;
        }
        const onChange = () => {
          if (pc.iceGatheringState === "complete") {
            pc.removeEventListener("icegatheringstatechange", onChange);
            resolve();
          }
        };
        pc.addEventListener("icegatheringstatechange", onChange);
      });
      const response = await fetch(`/api/whep/${CAMERA}`, {
        method: "POST",
        headers: { "Content-Type": "application/sdp" },
        body: pc.localDescription?.sdp ?? "",
      });
      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        throw new Error(detail || `WHEP signaling failed (${response.status})`);
      }
      const answer = await response.text();
      await pc.setRemoteDescription({ type: "answer", sdp: answer });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }, [stop]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void start();
    }, 0);
    return () => {
      window.clearTimeout(timer);
      stop();
    };
  }, [start, stop]);

  useEffect(() => {
    if (!detectOn) {
      return;
    }
    const source = new EventSource(`/api/internal/live/detections?camera=${CAMERA}&fps=2`);
    source.onopen = () => setBoxes([]);
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as { boxes?: LiveBox[] };
        if (Array.isArray(payload.boxes)) {
          setBoxes(payload.boxes);
        }
      } catch {
        // ignore a torn frame; EventSource keeps streaming
      }
    };
    return () => {
      source.close();
    };
  }, [detectOn]);

  useEffect(() => {
    if (status !== "live") {
      return;
    }
    const timer = setInterval(() => {
      pcRef.current
        ?.getStats()
        .then((report) => {
          report.forEach((entry) => {
            if (entry.type === "inbound-rtp" && "kind" in entry && entry.kind === "video") {
              const stats = entry as RTCInboundRtpStreamStats & { framesPerSecond?: number };
              if (typeof stats.framesPerSecond === "number") {
                setFps(stats.framesPerSecond);
              }
            }
          });
        })
        .catch(() => undefined);
    }, 1000);
    return () => clearInterval(timer);
  }, [status]);

  const visibleBoxes = useMemo(() => suppressOverlaps(boxes), [boxes]);

  return (
    <Card className="flex h-full min-h-0 flex-col overflow-hidden">
      <CardHeader className="shrink-0">
        <CardTitle className="flex items-center gap-2">
          <Video className="size-4" /> Camera feed
        </CardTitle>
        <CardDescription>
          WHEP playback from MediaMTX. Turn on detection to overlay live YOLO boxes.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
        <VideoStage
          aspect={CAMERA_ASPECT}
          className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl bg-muted/40"
        >
          {(rect) => (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute bg-black object-contain"
                style={rect}
              />
              {detectOn && visibleBoxes.length > 0 && (
                <div className="pointer-events-none absolute" style={rect}>
                  {visibleBoxes.map((box, index) => (
                    <div
                      key={`${index}-${box.x1}-${box.y1}`}
                      className="absolute border-2 border-sky-400/90 bg-sky-400/10"
                      style={{
                        left: `${box.x1 * 100}%`,
                        top: `${box.y1 * 100}%`,
                        width: `${(box.x2 - box.x1) * 100}%`,
                        height: `${(box.y2 - box.y1) * 100}%`,
                      }}
                    >
                      <span className="absolute -top-4 left-0 whitespace-nowrap rounded-sm bg-sky-500/90 px-1 text-[10px] font-medium text-white">
                        {box.label} {(box.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </VideoStage>
        {status === "error" && error && (
          <Alert variant="destructive" className="shrink-0">
            <CircleAlert />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex shrink-0 flex-wrap items-center gap-3">
        <Badge variant={status === "live" ? "default" : "outline"} className="gap-1.5">
          {status === "live" ? (
            <Radio className="size-3.5 animate-pulse" />
          ) : status === "connecting" ? (
            <LoaderCircle className="size-3.5 animate-spin" />
          ) : (
            <CircleAlert className="size-3.5" />
          )}
          {status === "live" ? "LIVE" : status === "connecting" ? "CONNECTING…" : "ERROR"}
        </Badge>
        <span className="text-xs text-muted-foreground">{CAMERA}</span>
        <span className="text-xs text-muted-foreground">H.264 · 1280×720 · target 25 fps</span>
        <div className="ml-auto flex items-center gap-2">
          <Label htmlFor="live-detection" className="flex items-center gap-1.5 text-xs">
            <Scan className="size-3.5" />
            Detection
          </Label>
          <Switch
            id="live-detection"
            checked={detectOn}
            onCheckedChange={setDetectOn}
            aria-label="Toggle live detection overlay"
          />
          {detectOn && (
            <Badge variant="secondary" className="tabular-nums">
              {boxes.length > 0 ? `${boxes.length} box(es)` : "detecting…"}
            </Badge>
          )}
          <span className="min-w-14 text-right text-xs tabular-nums text-muted-foreground">
            {fps !== null ? `${fps.toFixed(1)} fps` : "— fps"}
          </span>
          {status === "error" && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => {
                setStatus("connecting");
                setError(null);
                setFps(null);
                void start();
              }}
            >
              Reconnect
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}
