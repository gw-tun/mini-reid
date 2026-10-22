"use client";

import { Activity, Camera, Cpu, Database, Film, Scan, Users } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type Segment = { name: string; sizeBytes: number; modifiedAt: string };
type SegmentList = { segments: Segment[] };
type IndexStatus = {
  status: "indexed" | "processing" | "queued" | "failed" | "unknown";
  detections: number;
  embeddings: number;
};

const CAMERA = "cam01";
const RECENT_LIMIT = 8;

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export function DashboardOverview() {
  const [segments, setSegments] = useState<Segment[] | null>(null);
  const [indexStatus, setIndexStatus] = useState<Record<string, IndexStatus>>({});
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetch(`/api/recordings?camera=${CAMERA}`, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`segment list failed (${response.status})`);
        }
        return (await response.json()) as SegmentList;
      })
      .then((data) => {
        if (!cancelled) {
          setSegments(data.segments);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setFailed(err instanceof Error ? err.message : String(err));
        }
      });
    void fetch(`/api/internal/records/index-status?camera_key=${CAMERA}`, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`index status failed (${response.status})`);
        }
        return (await response.json()) as { segments: Record<string, IndexStatus> };
      })
      .then((data) => {
        if (!cancelled) {
          setIndexStatus(data.segments);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setIndexStatus({});
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const total = segments?.length ?? 0;
  const indexedCount = Object.values(indexStatus).filter(
    (entry) => entry.status === "indexed",
  ).length;
  const queuedCount = Object.values(indexStatus).filter(
    (entry) => entry.status === "queued" || entry.status === "processing",
  ).length;

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="flex shrink-0 flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Recording, detection and cross-record search status for {CAMERA}.
          </p>
        </div>
        <div className="flex gap-2">
          <Button asChild variant="outline" size="sm">
            <Link href="/live">Open live view</Link>
          </Button>
          <Button asChild size="sm">
            <Link href="/playback">Review playback</Link>
          </Button>
        </div>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-4 pr-2">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-1.5">
              <Film className="size-4" /> Recordings
            </CardDescription>
            <CardTitle className="text-2xl tabular-nums">
              {segments === null ? <Skeleton className="h-7 w-14" /> : total}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {CAMERA} segments available for review
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-1.5">
              <Database className="size-4" /> Indexed
            </CardDescription>
            <CardTitle className="text-2xl tabular-nums">
              {segments === null ? <Skeleton className="h-7 w-14" /> : indexedCount}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {indexedCount} indexed · {queuedCount} waiting in the worker queue
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-1.5">
              <Scan className="size-4" /> Detection engine
            </CardDescription>
            <CardTitle className="text-2xl">YOLO 11m</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center gap-2 text-xs text-muted-foreground">
            <Cpu className="size-3.5" /> CPU inference · live overlay + batch index
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription className="flex items-center gap-1.5">
              <Users className="size-4" /> ReID
            </CardDescription>
            <CardTitle className="text-2xl">FastReID</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center gap-2 text-xs text-muted-foreground">
            <Activity className="size-3.5" /> Embeddings power cross-record search
          </CardContent>
        </Card>
      </div>

      {failed && (
        <p className="text-sm text-destructive">Failed to load recordings: {failed}</p>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Recent recordings</CardTitle>
          <CardDescription>
            Newest segments first. Indexed segments expose green detection boxes in playback.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {segments === null ? (
            <div className="flex flex-col gap-2">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          ) : segments.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No recordings in <code>data/recordings/{CAMERA}</code> yet.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Segment</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Recorded</TableHead>
                  <TableHead>Index</TableHead>
                  <TableHead className="text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {segments.slice(0, RECENT_LIMIT).map((segment) => (
                  <TableRow key={segment.name}>
                    <TableCell className="font-medium">{segment.name}</TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">
                      {formatSize(segment.sizeBytes)}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {new Date(segment.modifiedAt).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      {indexStatus[segment.name]?.status === "indexed" ? (
                        <Badge variant="secondary">
                          indexed · {indexStatus[segment.name].detections}
                        </Badge>
                      ) : indexStatus[segment.name]?.status === "processing" ? (
                        <Badge variant="outline">processing</Badge>
                      ) : indexStatus[segment.name]?.status === "queued" ? (
                        <Badge
                          variant="outline"
                          className="border-amber-300 text-amber-700 dark:border-amber-800 dark:text-amber-400"
                        >
                          queued
                        </Badge>
                      ) : indexStatus[segment.name]?.status === "failed" ? (
                        <Badge variant="destructive">failed</Badge>
                      ) : (
                        <Badge variant="outline">not indexed</Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button asChild size="sm" variant="ghost">
                        <Link
                          href={`/playback?camera=${CAMERA}&segment=${encodeURIComponent(segment.name)}`}
                        >
                          <Camera className="size-4" />
                          Open
                        </Link>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
        </div>
      </ScrollArea>
    </div>
  );
}
