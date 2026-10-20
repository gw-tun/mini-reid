import Link from "next/link";
import { Suspense } from "react";

import { Crop, Problem } from "@/components/crop";
import { clock, getTracks, type PersonTrack } from "@/lib/api";

/** Every indexed person track, grouped by camera. Pick one to find similar people. */
export default function Home() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">Person tracks</h1>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        One card is one person followed inside one camera. Pick a person to find them in
        other tracks and cameras.
      </p>
      <Suspense fallback={<p className="mt-8 text-sm text-zinc-500">Loading tracks…</p>}>
        <TrackGrid />
      </Suspense>
    </main>
  );
}

async function TrackGrid() {
  let tracks: PersonTrack[];
  try {
    tracks = await getTracks();
  } catch (error) {
    return <div className="mt-8"><Problem error={error} /></div>;
  }
  if (tracks.length === 0) {
    return (
      <p className="mt-8 text-sm text-zinc-600 dark:text-zinc-400">
        No person tracks yet. Index a recording first:{" "}
        <code>.venv/bin/python scripts/index_clip.py VIDEO --device mps</code>
      </p>
    );
  }
  const byCamera = Map.groupBy(tracks, (t) => t.camera_key);
  return (
    <div className="mt-6 space-y-10">
      {[...byCamera].map(([camera, list]) => (
        <section key={camera}>
          <h2 className="text-lg font-medium">
            {camera} <span className="text-sm font-normal text-zinc-500">· {list.length} tracks</span>
          </h2>
          <ul className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {list
              .sort((a, b) => a.started_at.localeCompare(b.started_at))
              .map((t) => (
                <li key={t.track_id}>
                  <Link
                    href={`/tracks/${t.track_id}`}
                    className="block rounded-lg border border-zinc-200 p-2 hover:border-zinc-400 dark:border-zinc-800 dark:hover:border-zinc-600"
                  >
                    <div className="flex justify-center">
                      <Crop uri={t.best_crop_uri} alt={t.local_track_key} />
                    </div>
                    <p className="mt-2 truncate text-sm font-medium">{t.local_track_key}</p>
                    <p className="text-xs text-zinc-500">
                      {clock(t.started_at)} · {t.seconds ?? "?"} s · {t.observations} boxes
                    </p>
                  </Link>
                </li>
              ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
