"use client";

import { CircleAlert, Film, ListVideo, RefreshCw, Scan, Target } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { VideoStage } from "@/components/video-stage";

type Segment = { name: string; sizeBytes: number; modifiedAt: string };

type Detection = {
  detection_id: number;
  track_db_id: number | null;
  track_label: string | null;
  class_name: string;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  detector_score: number | null;
  offset_seconds: number | null;
  has_embedding: boolean;
};

type DetectionsPayload = {
  video: { width: number; height: number; fps: number; duration_seconds: number | null };
  detections: Detection[];
};

type IndexStatus = {
  status: "indexed" | "processing" | "queued" | "failed" | "unknown";
  detections: number;
  embeddings: number;
};

const DEFAULT_CAMERA = "cam01";
const TIME_WINDOW = 0.75;
const MAX_STALE_SECONDS = 1.25;

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / 1024).toFixed(0)} KB`;
}

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
function suppressOverlaps(detections: Detection[], threshold = 0.2): Detection[] {
  const sorted = [...detections].sort(
    (left, right) => (right.detector_score ?? 0) - (left.detector_score ?? 0),
  );
  const kept: Detection[] = [];
  for (const candidate of sorted) {
    const overlaps = kept.some(
      (existing) => intersectionOverUnion(existing.bbox, candidate.bbox) > threshold,
    );
    if (!overlaps) {
      kept.push(candidate);
    }
  }
  return kept;
}

/** Keep one box per person (track) so stacked detections do not clutter the frame. */
function dedupeDetections(detections: Detection[]): Detection[] {
  const bestByTrack = new Map<string, Detection>();
  const untracked: Detection[] = [];
  for (const detection of detections) {
    const key =
      detection.track_db_id !== null
        ? `track-${detection.track_db_id}`
        : detection.track_label
          ? `label-${detection.track_label}`
          : null;
    if (key === null) {
      untracked.push(detection);
      continue;
    }
    const current = bestByTrack.get(key);
    if (!current || (detection.detector_score ?? 0) > (current.detector_score ?? 0)) {
      bestByTrack.set(key, detection);
    }
  }
  return [...bestByTrack.values(), ...suppressOverlaps(untracked)];
}

function statusBadge(status: IndexStatus | undefined) {
  if (!status || status.status === "unknown") {
    return <Badge variant="outline">not indexed</Badge>;
  }
  if (status.status === "indexed") {
    return (
      <Badge variant="secondary" className="gap-1">
        indexed · {status.detections}
      </Badge>
    );
  }
  if (status.status === "processing") {
    return <Badge variant="outline">processing</Badge>;
  }
  if (status.status === "queued") {
    return <Badge variant="outline" className="border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400">
      queued
    </Badge>;
  }
  return <Badge variant="destructive">failed</Badge>;
}

export default function PlaybackClient() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const seekedRef = useRef(false);
  const pendingSeekRef = useRef<number | null>(null);
  const autoSeekDoneRef = useRef(false);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [detectionPayload, setDetectionPayload] = useState<{
    segment: string;
    data: DetectionsPayload;
  } | null>(null);
  const [indexState, setIndexState] = useState<"loading" | "indexed" | "unindexed" | "unknown">(
    "loading",
  );
  const [indexStatus, setIndexStatus] = useState<Record<string, IndexStatus>>({});
  const [currentTime, setCurrentTime] = useState(0);
  const [contextDetection, setContextDetection] = useState<Detection | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [cameraKey, setCameraKey] = useState(DEFAULT_CAMERA);

  const fetchSegments = useCallback(async (camera: string) => {
    const response = await fetch(`/api/recordings?camera=${camera}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`segment list failed (${response.status})`);
    }
    const data = (await response.json()) as { segments: Segment[] };
    return data.segments;
  }, []);

  const fetchIndexStatus = useCallback(async (camera: string) => {
    const response = await fetch(
      `/api/internal/records/index-status?camera_key=${encodeURIComponent(camera)}`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      throw new Error(`index status failed (${response.status})`);
    }
    const data = (await response.json()) as { segments: Record<string, IndexStatus> };
    return data.segments;
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const camera = params.get("camera") ?? DEFAULT_CAMERA;
    const segment = params.get("segment");
    const t = Number.parseFloat(params.get("t") ?? "");
    void fetchSegments(camera)
      .then((items) => {
        setSegments(items);
        setState("ready");
        setCameraKey(camera);
        const names = items.map((item) => item.name);
        setIndexState("loading");
        setSelected(segment && names.includes(segment) ? segment : (names[0] ?? null));
        if (Number.isFinite(t) && t >= 0) {
          seekedRef.current = false;
          pendingSeekRef.current = t;
        }
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
        setState("error");
      });
    void fetchIndexStatus(camera)
      .then((statuses) => {
        setIndexStatus(statuses);
      })
      .catch(() => {
        setIndexStatus({});
      });
  }, [fetchSegments, fetchIndexStatus]);

  useEffect(() => {
    if (!selected) {
      return;
    }
    const camera = new URLSearchParams(window.location.search).get("camera") ?? DEFAULT_CAMERA;
    let cancelled = false;
    void (async () => {
      let data: DetectionsPayload | null = null;
      let status: "indexed" | "unindexed" | "unknown" = "unknown";
      try {
        const response = await fetch(
          `/api/internal/records/detections?camera=${encodeURIComponent(camera)}&segment=${encodeURIComponent(selected)}`,
          { cache: "no-store" },
        );
        if (response.ok) {
          data = (await response.json()) as DetectionsPayload;
          status = "indexed";
        } else if (response.status === 404) {
          status = "unindexed";
        }
      } catch {
        data = null;
      }
      if (!cancelled) {
        setIndexState(status);
        setDetectionPayload({
          segment: selected,
          data:
            data ?? {
              video: { width: 0, height: 0, fps: 0, duration_seconds: null },
              detections: [],
            },
        });
        if (!autoSeekDoneRef.current) {
          autoSeekDoneRef.current = true;
          const hasTimeParam = Number.isFinite(
            Number.parseFloat(new URLSearchParams(window.location.search).get("t") ?? ""),
          );
          const first = data?.detections.find((item) => item.offset_seconds !== null);
          if (!hasTimeParam && first?.offset_seconds != null) {
            const video = videoRef.current;
            if (video && video.readyState >= 1) {
              const duration = Number.isFinite(video.duration) ? video.duration : first.offset_seconds;
              video.currentTime = Math.min(first.offset_seconds, Math.max(duration - 0.1, 0));
              setCurrentTime(first.offset_seconds);
            } else {
              pendingSeekRef.current = first.offset_seconds;
            }
          }
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const activePayload =
    detectionPayload && detectionPayload.segment === selected ? detectionPayload.data : null;
  const videoMeta = activePayload?.video ?? null;
  const detections = activePayload?.detections ?? [];

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    const target = pendingSeekRef.current;
    if (target !== null && !seekedRef.current && Number.isFinite(target)) {
      seekedRef.current = true;
      video.currentTime = Math.min(target, Math.max(video.duration - 0.1, 0));
      pendingSeekRef.current = null;
    }
  };

  const visibleDetections = useMemo(() => {
    if (!videoMeta || videoMeta.width <= 0) {
      return [];
    }
    const stamped = detections.filter(
      (item): item is Detection & { offset_seconds: number } => item.offset_seconds !== null,
    );
    if (stamped.length === 0) {
      return [];
    }
    // The worker samples every ~0.5s, so the same person shows up as a new track in
    // every sample. Show the closest sampling instant only, otherwise one person
    // appears as several boxes side by side.
    let nearest = stamped[0].offset_seconds;
    for (const item of stamped) {
      if (Math.abs(item.offset_seconds - currentTime) < Math.abs(nearest - currentTime)) {
        nearest = item.offset_seconds;
      }
    }
    if (Math.abs(nearest - currentTime) > MAX_STALE_SECONDS) {
      return [];
    }
    const sameInstant = stamped.filter((item) => Math.abs(item.offset_seconds - nearest) <= 0.05);
    const inWindow = sameInstant.length > 0
      ? sameInstant
      : stamped.filter((item) => Math.abs(item.offset_seconds - currentTime) <= TIME_WINDOW);
    return dedupeDetections(inWindow);
  }, [currentTime, detections, videoMeta]);

  const seekTo = (seconds: number) => {
    const video = videoRef.current;
    if (!video) return;
    const duration = Number.isFinite(video.duration) ? video.duration : seconds;
    video.currentTime = Math.min(seconds, Math.max(duration - 0.1, 0));
    setCurrentTime(seconds);
  };

  const openDialog = (detection: Detection, event: React.MouseEvent) => {
    event.preventDefault();
    setContextDetection(detection);
  };

  const startCrossRecordSearch = () => {
    if (!contextDetection) return;
    const camera = new URLSearchParams(window.location.search).get("camera") ?? DEFAULT_CAMERA;
    const params = new URLSearchParams({
      camera,
      segment: selected ?? "",
      detection: String(contextDetection.detection_id),
    });
    if (contextDetection.track_db_id !== null) {
      params.set("track_db_id", String(contextDetection.track_db_id));
    }
    router.push(`/record-search?${params.toString()}`);
  };

  const selectSegment = (name: string) => {
    seekedRef.current = false;
    pendingSeekRef.current = null;
    setIndexState("loading");
    setSelected(name);
    setListOpen(false);
  };

  return (
    <Card className="flex h-full min-h-0 flex-col">
      <CardHeader className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Film className="size-4" /> Playback Review
          </CardTitle>
          <CardDescription>
            Pick a segment, jump to a detection, then click a green box to search this person
            across every indexed recording.
          </CardDescription>
        </div>
        <div className="flex items-center gap-2">
          <Sheet open={listOpen} onOpenChange={setListOpen}>
            <SheetTrigger asChild>
              <Button type="button" variant="outline" size="sm">
                <ListVideo className="size-4" />
                Recordings ({segments.length})
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="flex w-full flex-col gap-3 sm:max-w-md">
              <SheetHeader>
                <SheetTitle>Recordings</SheetTitle>
                <SheetDescription>
                  {cameraKey} · newest first. The badge shows whether the worker already indexed
                  the segment, is processing it, or is still waiting in the queue.
                </SheetDescription>
              </SheetHeader>
              <ScrollArea className="min-h-0 flex-1">
                <div className="divide-y pr-2">
                  {segments.length === 0 ? (
                    <p className="py-4 text-sm text-muted-foreground">No recordings yet.</p>
                  ) : (
                    segments.map((segment) => (
                      <button
                        key={segment.name}
                        type="button"
                        onClick={() => selectSegment(segment.name)}
                        className={`flex w-full flex-col gap-1 px-2 py-2.5 text-left transition-colors ${
                          selected === segment.name
                            ? "bg-primary/10"
                            : "hover:bg-muted"
                        }`}
                      >
                        <span className="flex items-center justify-between gap-2">
                          <span className="truncate text-sm font-medium">{segment.name}</span>
                          {statusBadge(indexStatus[segment.name])}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {formatSize(segment.sizeBytes)} ·{" "}
                          {new Date(segment.modifiedAt).toLocaleString()}
                        </span>
                      </button>
                    ))
                  )}
                </div>
              </ScrollArea>
            </SheetContent>
          </Sheet>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              setState("loading");
              setError(null);
              void fetchSegments(cameraKey)
                .then((items) => {
                  setSegments(items);
                  setState("ready");
                })
                .catch((err: unknown) => {
                  setError(err instanceof Error ? err.message : String(err));
                  setState("error");
                });
              void fetchIndexStatus(cameraKey)
                .then((statuses) => {
                  setIndexStatus(statuses);
                })
                .catch(() => {
                  setIndexStatus({});
                });
            }}
          >
            <RefreshCw className="size-4" />
            Refresh
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
        {state === "loading" && <p className="text-sm text-muted-foreground">Loading segments…</p>}
        {state === "error" && (
          <p className="text-sm text-destructive">{error ?? "Failed to load"}</p>
        )}

        {state === "ready" && segments.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No recordings in <code>data/recordings/{cameraKey}</code> yet.
          </p>
        )}

        {state === "ready" && segments.length > 0 && (
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            {selected ? (
              <>
                <VideoStage
                  aspect={
                    videoMeta && videoMeta.height > 0
                      ? videoMeta.width / videoMeta.height
                      : 16 / 9
                  }
                  className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl border bg-muted/40"
                >
                  {(rect) => (
                    <>
                      <video
                        key={selected}
                        ref={videoRef}
                        controls
                        preload="metadata"
                        className="absolute bg-black object-contain"
                        style={rect}
                        src={`/api/recordings/${encodeURIComponent(selected)}?camera=${cameraKey}`}
                        onLoadedMetadata={handleLoadedMetadata}
                        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                      >
                        <track kind="captions" />
                      </video>
                      {videoMeta && videoMeta.width > 0 && (
                        <div className="pointer-events-none absolute" style={rect}>
                          {visibleDetections.map((detection) => {
                            const { width, height } = videoMeta;
                            return (
                              <div
                                key={detection.detection_id}
                                className="pointer-events-auto absolute cursor-context-menu border-2 border-emerald-400/90 bg-emerald-400/10"
                                style={{
                                  left: `${(detection.bbox.x1 / width) * 100}%`,
                                  top: `${(detection.bbox.y1 / height) * 100}%`,
                                  width: `${((detection.bbox.x2 - detection.bbox.x1) / width) * 100}%`,
                                  height: `${((detection.bbox.y2 - detection.bbox.y1) / height) * 100}%`,
                                }}
                                title={`${detection.class_name} ${detection.track_label ?? ""} (click or right-click for cross-record search)`}
                                onClick={(event) => openDialog(detection, event)}
                                onContextMenu={(event) => openDialog(detection, event)}
                              >
                                <span className="absolute -top-5 left-0 rounded-sm bg-emerald-500/90 px-1 text-[10px] font-medium text-white">
                                  {detection.track_label ?? detection.class_name}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </>
                  )}
                </VideoStage>

                {indexState === "unindexed" && (
                  <Alert>
                    <CircleAlert />
                    <AlertTitle>Not indexed yet</AlertTitle>
                    <AlertDescription>
                      {indexStatus[selected]?.status === "queued"
                        ? "This segment is still waiting in the worker queue — green boxes appear once it is indexed."
                        : "This segment is not indexed by the worker yet — green boxes appear once indexing completes."}
                    </AlertDescription>
                  </Alert>
                )}

                {indexState === "indexed" && detections.length > 0 && (
                  <div className="flex shrink-0 flex-wrap items-center gap-1.5">
                    <span className="text-xs text-muted-foreground">Jump to detection:</span>
                    {detections.map((detection) => (
                      <Button
                        key={detection.detection_id}
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={detection.offset_seconds === null}
                        onClick={() => seekTo(detection.offset_seconds ?? 0)}
                        className="h-7 border-emerald-300 px-2 text-xs text-emerald-700 hover:bg-emerald-50 dark:border-emerald-800 dark:text-emerald-400 dark:hover:bg-emerald-950/40"
                      >
                        {(detection.offset_seconds ?? 0).toFixed(1)}s{" "}
                        {detection.track_label ?? detection.class_name}
                      </Button>
                    ))}
                  </div>
                )}

                <p className="flex shrink-0 flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <span>{selected}</span>
                  {statusBadge(indexStatus[selected])}
                  <span>· click a green box for cross-record search</span>
                  {detections.length > 0 && (
                    <span>· {detections.length} indexed detection(s)</span>
                  )}
                </p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                Select a recording from the Recordings list.
              </p>
            )}
          </div>
        )}
      </CardContent>
      <CardFooter className="flex items-center gap-2 text-xs text-muted-foreground">
        <Scan className="size-3.5" />
        Detections are produced by the worker pipeline; indexed segments expose green boxes.
        <Separator orientation="vertical" className="mx-1 h-4" />
        <Target className="size-3.5" />
        Cross-record search uses stored ReID embeddings.
      </CardFooter>

      <Dialog
        open={contextDetection !== null}
        onOpenChange={(open) => {
          if (!open) {
            setContextDetection(null);
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Detection detail</DialogTitle>
            <DialogDescription>
              {contextDetection
                ? `${contextDetection.class_name} · ${contextDetection.track_label ?? "untracked"} · ${
                    contextDetection.offset_seconds !== null
                      ? `${contextDetection.offset_seconds.toFixed(2)}s`
                      : "time unknown"
                  }`
                : ""}
            </DialogDescription>
          </DialogHeader>
          {contextDetection && (
            <>
              <img
                src={`/api/internal/detections/${contextDetection.detection_id}/crop`}
                alt="Detection crop"
                className="mx-auto max-h-52 w-auto rounded-md border bg-black object-contain"
              />
              {contextDetection.has_embedding ? (
                <Alert>
                  <Target />
                  <AlertTitle>Cross Record Search</AlertTitle>
                  <AlertDescription className="flex flex-col gap-2">
                    <span>Search this person across all indexed recordings?</span>
                    <div className="flex gap-2">
                      <Button type="button" size="sm" onClick={startCrossRecordSearch}>
                        Search across records
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={() => setContextDetection(null)}
                      >
                        Cancel
                      </Button>
                    </div>
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert>
                  <CircleAlert />
                  <AlertTitle>No embedding</AlertTitle>
                  <AlertDescription>
                    No stored embedding for this detection — cross-record search is unavailable.
                  </AlertDescription>
                </Alert>
              )}
            </>
          )}
          <DialogFooter>
            <Badge variant="outline" className="mr-auto">
              detection #{contextDetection?.detection_id ?? "—"}
            </Badge>
            <Button type="button" variant="outline" onClick={() => setContextDetection(null)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
