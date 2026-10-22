/**
 * Types and URLs for the recordings page and "Search similar". Safe in the
 * browser: nothing here talks to the API from the server. Types mirror
 * services/api/app/schemas.py.
 */
import type { PersonTrack } from "@/lib/api";

export type Recording = {
  path: string;
  camera: string;
  recorded_from: string | null;
  seconds: number | null;
  /** share, snippet (both full resolution) or api (960 px). */
  via: string | null;
  width: number | null;
  height: number | null;
  processed: boolean;
  people: number | null;
  rank: number | null;
  score: number | null;
  semantic_va_clip_id: number | null;
};

export type RecordingGroup = {
  at: string | null;
  text: string | null;
  predicate: string | null;
  recordings: Recording[];
};

export type RecordingPeople = {
  path: string;
  camera: string;
  recorded_from: string | null;
  processed_now: boolean;
  people: PersonTrack[];
};

export type RecordingMatch = {
  rank: number;
  track: PersonTrack;
  /** Cosine of mean barcodes, one model. Not a probability. */
  similarity: number;
  /** Same camera, overlapping the chosen person in time: certainly someone else. */
  same_time: boolean;
  path: string;
  /** Seconds into `path` where this person first appears. */
  offset: number;
};

export type RecordingSearch = {
  search_id: string;
  status: "queued" | "running" | "complete" | "failed";
  stage: string | null;
  done: number;
  total: number;
  error: string | null;
  model_key: string;
  query_track_id: number;
  results: RecordingMatch[];
};

const encodePath = (path: string) => path.split("/").map(encodeURIComponent).join("/");

/** The recording for a <video>: through this app's /api proxy, with seeking (Range). */
export const videoUrl = (path: string) => `/api/recordings/video/${encodePath(path)}`;

/** This app's page for searching similar people from one recording. */
export const searchPage = (path: string) => `/recordings/${encodePath(path)}`;

/** "2018-03-13" -- recording times are the file name's clock, labelled UTC. */
export const day = (iso: string) => new Date(iso).toISOString().slice(0, 10);
