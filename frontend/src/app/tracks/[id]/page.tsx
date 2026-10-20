import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";

import { Crop, Problem } from "@/components/crop";
import {
  ApiError,
  MODEL_NAMES,
  clock,
  getModels,
  getSimilar,
  getTrack,
  type PersonTrackDetail,
  type ReIDModelInfo,
  type SimilarTracksResponse,
} from "@/lib/api";

/** One person track, and the most similar tracks according to each ReID model. */
export default function TrackPage({ params }: PageProps<"/tracks/[id]">) {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8">
      <Link href="/tracks" className="text-sm text-zinc-500 hover:underline">
        ← All person tracks
      </Link>
      <Suspense fallback={<p className="mt-8 text-sm text-zinc-500">Loading…</p>}>
        {params.then(({ id }) => (
          <TrackView id={id} />
        ))}
      </Suspense>
    </main>
  );
}

async function TrackView({ id }: { id: string }) {
  if (!/^\d+$/.test(id)) notFound();
  let track: PersonTrackDetail;
  let models: ReIDModelInfo[];
  try {
    [track, models] = await Promise.all([getTrack(id), getModels()]);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) notFound();
    return <div className="mt-8"><Problem error={error} /></div>;
  }
  const end = track.ended_at ? clock(track.ended_at) : "now";
  return (
    <>
      <h1 className="mt-4 text-2xl font-semibold tracking-tight">{track.local_track_key}</h1>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        Camera <b>{track.camera_key}</b> · {clock(track.started_at)}–{end} UTC ·{" "}
        {track.observations} boxes · {track.recording_uri ?? "live"}
      </p>
      <div className="mt-4 flex flex-wrap gap-3">
        {track.crops.length === 0 && <Crop uri={null} alt="" className="h-48 w-28" />}
        {track.crops.map((c) => (
          <figure key={c.uri} className="text-center">
            <Crop uri={c.uri} alt={`${track.local_track_key} at ${clock(c.observed_at)}`} className="h-48" />
            <figcaption className="mt-1 text-xs text-zinc-500">
              {clock(c.observed_at)} · quality {c.quality.toFixed(2)}
            </figcaption>
          </figure>
        ))}
      </div>

      <h2 className="mt-10 text-lg font-medium">Similar people</h2>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        Each model compares this person&apos;s barcode with every other track. A score is only
        comparable with scores from the <i>same</i> model. A red row was on screen at the same
        time on the same camera, so it is certainly someone else: it shows how high a wrong
        match can score.
      </p>
      {track.models.length === 0 ? (
        <p className="mt-4 text-sm text-zinc-500">This track has no usable crop, so no barcode.</p>
      ) : (
        <div className="mt-4 grid gap-6 md:grid-cols-2">
          {models
            .filter((m) => track.models.includes(m.model_key))
            .map((m) => (
              <section key={m.model_key} className="rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
                <h3 className="font-medium">
                  {MODEL_NAMES[m.model_key] ?? m.model_key}{" "}
                  <span className="text-xs font-normal text-zinc-500">
                    {m.backend} · {m.feature_dim} numbers · {m.tracks} tracks
                  </span>
                </h3>
                <Suspense fallback={<p className="mt-3 text-sm text-zinc-500">Searching…</p>}>
                  <SimilarList id={id} modelKey={m.model_key} />
                </Suspense>
              </section>
            ))}
        </div>
      )}
    </>
  );
}

async function SimilarList({ id, modelKey }: { id: string; modelKey: string }) {
  let similar: SimilarTracksResponse;
  try {
    similar = await getSimilar(id, modelKey);
  } catch (error) {
    return <div className="mt-3"><Problem error={error} /></div>;
  }
  if (similar.results.length === 0) {
    return <p className="mt-3 text-sm text-zinc-500">No other tracks with this model yet.</p>;
  }
  return (
    <ol className="mt-3 space-y-2">
      {similar.results.map(({ track: t, similarity, same_time }) => (
        <li key={t.track_id}>
          <Link
            href={`/tracks/${t.track_id}`}
            className={`flex items-center gap-3 rounded p-2 hover:bg-zinc-50 dark:hover:bg-zinc-900 ${
              same_time ? "bg-red-50 dark:bg-red-950/40" : ""
            }`}
          >
            <Crop uri={t.best_crop_uri} alt={t.local_track_key} className="h-20" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{t.local_track_key}</p>
              <p className="text-xs text-zinc-500">
                {t.camera_key} · {clock(t.started_at)} UTC
              </p>
              <div className="mt-1 flex items-center gap-2">
                <div className="h-1.5 flex-1 rounded bg-zinc-200 dark:bg-zinc-800">
                  <div
                    className={`h-1.5 rounded ${same_time ? "bg-red-500" : "bg-emerald-500"}`}
                    style={{ width: `${Math.max(0, similarity) * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-xs">{similarity.toFixed(3)}</span>
              </div>
              {same_time && (
                <p className="mt-0.5 text-xs text-red-700 dark:text-red-400">
                  on screen together: a different person
                </p>
              )}
            </div>
          </Link>
        </li>
      ))}
    </ol>
  );
}
