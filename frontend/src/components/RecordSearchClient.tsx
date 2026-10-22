"use client";

import { CircleAlert, RefreshCw, Search, Target, Users } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

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
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type QueryInfo = {
  camera_key: string;
  segment: string | null;
  track_db_id: number;
  track_label: string | null;
  class_name: string;
  embedding_samples: number;
};

type Match = {
  camera_key: string;
  segment: string;
  track_db_id: number | null;
  track_label: string | null;
  class_name: string;
  score: number;
  detection_id: number | null;
  offset_seconds: number | null;
};

type SearchResult = {
  query: QueryInfo;
  min_similarity: number;
  matches: Match[];
};

function groupByRecord(matches: Match[]): { key: string; best: Match; count: number }[] {
  const groups = new Map<string, { best: Match; count: number }>();
  for (const match of matches) {
    const key = `${match.camera_key}/${match.segment}`;
    const existing = groups.get(key);
    if (!existing || match.score > existing.best.score) {
      groups.set(key, { best: match, count: (existing?.count ?? 0) + 1 });
    } else {
      existing.count += 1;
    }
  }
  return [...groups.entries()]
    .map(([key, value]) => ({ key, best: value.best, count: value.count }))
    .sort((left, right) => right.best.score - left.best.score);
}

type QueryOutcome =
  | { ok: true; result: SearchResult; detection: string | null }
  | { ok: false; message: string };

type QueryParams = {
  camera: string | null;
  segment: string | null;
  detection: string | null;
  trackDbId: string | null;
};

const MISSING_QUERY_MESSAGE =
  "Open a detection box in Playback Review and choose “Search across records”.";

export default function RecordSearchClient() {
  const paramsRef = useRef<QueryParams>({
    camera: null,
    segment: null,
    detection: null,
    trackDbId: null,
  });
  const [result, setResult] = useState<SearchResult | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [sourceCrop, setSourceCrop] = useState<string | null>(null);

  const fetchOutcome = useCallback(async (): Promise<QueryOutcome> => {
    const { camera, segment, detection, trackDbId } = paramsRef.current;
    if (!camera || (!detection && !trackDbId)) {
      return { ok: false, message: MISSING_QUERY_MESSAGE };
    }
    try {
      const response = await fetch("/api/internal/record-search/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          camera_key: camera,
          segment: segment ?? undefined,
          detection_id: detection ? Number(detection) : undefined,
          track_db_id: trackDbId ? Number(trackDbId) : undefined,
        }),
      });
      if (!response.ok) {
        const detail = (await response.json().catch(() => null)) as { detail?: string } | null;
        return { ok: false, message: detail?.detail ?? `search failed (${response.status})` };
      }
      const data = (await response.json()) as SearchResult;
      return { ok: true, result: data, detection };
    } catch (err) {
      return { ok: false, message: err instanceof Error ? err.message : String(err) };
    }
  }, []);

  const applyOutcome = useCallback((outcome: QueryOutcome) => {
    if (!outcome.ok) {
      setError(outcome.message);
      setState("error");
      return;
    }
    setResult(outcome.result);
    if (outcome.detection) {
      setSourceCrop(`/api/internal/detections/${outcome.detection}/crop`);
    }
    setState("ready");
  }, []);

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    paramsRef.current = {
      camera: query.get("camera"),
      segment: query.get("segment"),
      detection: query.get("detection"),
      trackDbId: query.get("track_db_id"),
    };
    void fetchOutcome().then(applyOutcome);
  }, [fetchOutcome, applyOutcome]);

  const rerun = () => {
    setState("loading");
    setError(null);
    void fetchOutcome().then(applyOutcome);
  };

  const records = result ? groupByRecord(result.matches) : [];

  return (
    <Card className="flex h-full min-h-0 flex-col">
      <CardHeader className="shrink-0 flex flex-wrap items-start justify-between gap-3">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Search className="size-4" /> Cross Record Search
          </CardTitle>
          <CardDescription>
            ReID similarity search over every indexed recording for the queried person.
          </CardDescription>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={rerun}>
          <RefreshCw className="size-4" />
          Re-run
        </Button>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col">
        {state === "loading" && (
          <p className="text-sm text-muted-foreground">Searching indexed recordings…</p>
        )}
        {state === "error" && (
          <Alert variant="destructive">
            <CircleAlert />
            <AlertTitle>Search failed</AlertTitle>
            <AlertDescription>{error ?? "Search failed"}</AlertDescription>
          </Alert>
        )}

        {state === "ready" && result && (
          <ScrollArea className="min-h-0 flex-1 pr-2">
            <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-start gap-4 rounded-lg border p-4">
              {sourceCrop ? (
                <img
                  src={sourceCrop}
                  alt="Query detection crop"
                  className="h-28 w-20 rounded-md border bg-black object-cover"
                />
              ) : (
                <div className="h-28 w-20 rounded-md bg-muted" />
              )}
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold">
                  Query track{" "}
                  <code className="text-xs">{result.query.track_label ?? `#${result.query.track_db_id}`}</code>
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {result.query.class_name} · {result.query.embedding_samples} embedding sample(s) ·
                  threshold {result.min_similarity}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Source: {result.query.camera_key}/{result.query.segment ?? "unknown segment"}
                </p>
                <Button asChild size="sm" variant="link" className="mt-1 h-auto p-0">
                  <Link
                    href={`/playback?camera=${encodeURIComponent(result.query.camera_key)}${
                      result.query.segment ? `&segment=${encodeURIComponent(result.query.segment)}` : ""
                    }`}
                  >
                    ← Back to playback
                  </Link>
                </Button>
              </div>
              <Badge variant="secondary" className="gap-1.5">
                <Users className="size-3.5" />
                {records.length} record(s)
              </Badge>
            </div>

            {records.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No other records matched above the similarity threshold ({result.min_similarity}).
              </p>
            ) : (
              <>
                <p className="text-sm text-muted-foreground">
                  {records.length} record(s) contain this person (excluding the query track itself).
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Record</TableHead>
                      <TableHead>Best score</TableHead>
                      <TableHead>Tracks</TableHead>
                      <TableHead>Track</TableHead>
                      <TableHead>Offset</TableHead>
                      <TableHead className="text-right">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {records.map((record) => (
                      <TableRow key={record.key}>
                        <TableCell>
                          <div className="flex items-center gap-3">
                            {record.best.detection_id !== null ? (
                              <img
                                src={`/api/internal/detections/${record.best.detection_id}/crop`}
                                alt={`Match crop ${record.key}`}
                                className="h-14 w-10 rounded-md border bg-black object-cover"
                              />
                            ) : (
                              <div className="h-14 w-10 rounded-md bg-muted" />
                            )}
                            <span className="font-medium">
                              {record.best.camera_key}/{record.best.segment}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="tabular-nums">
                          {(record.best.score * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell className="tabular-nums">{record.count}</TableCell>
                        <TableCell>
                          {record.best.track_label ?? `#${record.best.track_db_id}`}
                        </TableCell>
                        <TableCell className="tabular-nums text-muted-foreground">
                          {record.best.offset_seconds !== null
                            ? `${record.best.offset_seconds.toFixed(1)}s`
                            : "—"}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button asChild size="sm">
                            <Link
                              href={`/playback?camera=${encodeURIComponent(
                                record.best.camera_key,
                              )}&segment=${encodeURIComponent(
                                record.best.segment,
                              )}${record.best.offset_seconds !== null ? `&t=${record.best.offset_seconds}` : ""}`}
                            >
                              Open record
                            </Link>
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </>
            )}
            </div>
          </ScrollArea>
        )}
      </CardContent>
      <CardFooter className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground">
        <Target className="size-3.5" />
        Scores come from stored ReID embeddings; the source track is excluded from results.
        <Separator orientation="vertical" className="mx-1 h-4" />
        Open a record to review the matched frame in playback.
      </CardFooter>
    </Card>
  );
}
