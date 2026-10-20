"use client";

/**
 * "Search similar" on one recording, in three steps:
 *
 *   1. POST /recordings/people   -- process the recording if needed (YOLO,
 *      tracking, crops, ReID barcodes into pgvector) and list its people;
 *   2. POST /recordings/searches -- for the person picked, process every other
 *      recording not processed yet, then rank everyone (polled: it can take a
 *      while the first time);
 *   3. the ranked people, each playable where they first appear.
 *
 * A similarity is a cosine between barcodes of ONE model: not a probability,
 * and not comparable between models.
 */

import { useCallback, useEffect, useState } from "react";

import { MODEL_NAMES, clock, cropUrl } from "@/lib/api";
import { day, videoUrl, type RecordingPeople, type RecordingSearch } from "@/lib/recordings";

const MODELS = ["msmt17_sbs_r50_ibn", "osnet_ain_x1_0_msmt17"];

async function call<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, body === undefined ? undefined : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("The mini-reid API can't be reached. Is it running on port 8000?");
  }
  if (!res.ok) {
    const detail = ((await res.json().catch(() => ({}))) as { detail?: unknown }).detail;
    throw new Error(typeof detail === "string" ? detail
      : res.status === 404 ? "Not found. If the API was started before this page existed, restart it."
      : `The API answered ${res.status}.`);
  }
  return res.json() as Promise<T>;
}

type Playing = { path: string; offset: number; label: string };

export function SimilarSearch({ path }: { path: string }) {
  const [people, setPeople] = useState<RecordingPeople | null>(null);
  const [chosen, setChosen] = useState<number | null>(null);
  const [model, setModel] = useState(MODELS[0]);
  const [search, setSearch] = useState<RecordingSearch | null>(null);
  const [playing, setPlaying] = useState<Playing>({ path, offset: 0, label: "this recording" });
  const [error, setError] = useState("");

  // Step 1: the people in this recording (processing it the first time).
  useEffect(() => {
    let live = true;
    call<RecordingPeople>("/recordings/people", { path })
      .then((p) => {
        if (!live) return;
        setPeople(p);
        if (p.people.length === 1) setChosen(p.people[0].track_id);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [path]);

  // Step 2: poll a running search once a second until it ends.
  const searchId = search?.search_id;
  const running = search?.status === "queued" || search?.status === "running";
  useEffect(() => {
    if (!searchId || !running) return;
    const timer = setInterval(() => {
      call<RecordingSearch>(`/recordings/searches/${searchId}`).then(setSearch).catch((e: Error) => setError(e.message));
    }, 1000);
    return () => clearInterval(timer);
  }, [searchId, running]);

  const start = useCallback(async () => {
    if (chosen == null) return;
    setError("");
    try {
      const { search_id } = await call<{ search_id: string }>("/recordings/searches", {
        track_id: chosen, model_key: model, top_k: 30,
      });
      setSearch({ search_id, status: "queued", stage: null, done: 0, total: 0, error: null,
        model_key: model, query_track_id: chosen, results: [] });
    } catch (e) {
      setError((e as Error).message);
    }
  }, [chosen, model]);

  const play = (p: Playing) => {
    setPlaying(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <>
      <div className="mt-4 grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div>
          {/* Keyed, so a new recording or start point is a new element that seeks once loaded. */}
          <video
            key={`${playing.path}@${playing.offset}`}
            src={videoUrl(playing.path)}
            controls
            autoPlay={playing.offset > 0}
            onLoadedMetadata={(e) => {
              e.currentTarget.currentTime = playing.offset;
            }}
            className="aspect-video w-full rounded bg-black"
          />
          <p className="mt-1 text-xs text-zinc-500">
            Playing {playing.label}: {playing.path}
            {playing.offset > 0 && <> from {playing.offset.toFixed(1)} s</>}
            {playing.path !== path && (
              <button onClick={() => setPlaying({ path, offset: 0, label: "this recording" })}
                className="ml-2 text-emerald-700 hover:underline dark:text-emerald-400">
                back to this recording
              </button>
            )}
          </p>
        </div>
        <div className="text-sm">
          <h1 className="text-xl font-semibold tracking-tight">Search similar</h1>
          <p className="mt-1 text-zinc-600 dark:text-zinc-400">
            {people ? (
              <>Camera <b>{people.camera}</b>
                {people.recorded_from && <> · {day(people.recorded_from)} {clock(people.recorded_from)}</>}</>
            ) : path}
          </p>
          {error && (
            <p className="mt-3 rounded border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
              {error}
            </p>
          )}
        </div>
      </div>

      <section className="mt-6">
        <h2 className="font-medium">1 · Who?</h2>
        {!people && !error && (
          <p className="mt-2 text-sm text-zinc-500">
            Processing this recording: YOLO → tracking → best crops → ReID barcodes → pgvector… The first time,
            the models load too, which can take a minute.
          </p>
        )}
        {people && (
          <p className="mt-1 text-xs text-zinc-500">
            {people.processed_now ? "Processed just now." : "Processed earlier."} {people.people.length} people with a
            usable crop. Pick one.
          </p>
        )}
        {people && people.people.length === 0 && (
          <p className="mt-2 text-sm text-zinc-500">
            No one here is big and clear enough to identify: the crop gate needs people at least 96 px tall. A
            recording downloaded at 960 px loses most small people; download it with <code>--source snippet</code>.
          </p>
        )}
        {people && people.people.length > 0 && (
          <>
            <div className="mt-2 flex flex-wrap gap-2">
              {people.people.map((p) => (
                <button
                  key={p.track_id}
                  onClick={() => setChosen(p.track_id)}
                  disabled={running}
                  aria-pressed={chosen === p.track_id}
                  title={`seen ${clock(p.started_at)} for ${p.seconds ?? "?"} s`}
                  className={`rounded border p-1 ${chosen === p.track_id ? "border-emerald-600 ring-2 ring-emerald-600" : "border-zinc-200 dark:border-zinc-700"}`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={cropUrl(p.best_crop_uri!)} alt={`person ${p.track_id}`} className="h-28 w-auto rounded object-contain" />
                </button>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
              <label className="flex items-center gap-1">
                <span className="text-zinc-500">Model</span>
                <select value={model} onChange={(e) => setModel(e.target.value)} disabled={running}
                  className="rounded border border-zinc-300 bg-transparent px-1 py-0.5 dark:border-zinc-700">
                  {MODELS.map((m) => <option key={m} value={m}>{MODEL_NAMES[m] ?? m}</option>)}
                </select>
              </label>
              <button onClick={start} disabled={chosen == null || running}
                className="rounded bg-emerald-600 px-3 py-1.5 font-medium text-white disabled:opacity-40">
                {chosen == null ? "Pick a person" : running ? "Searching…" : "Search similar"}
              </button>
            </div>
          </>
        )}
      </section>

      {search && running && (
        <section className="mt-6">
          <h2 className="font-medium">2 · Searching</h2>
          <p className="mt-1 text-sm text-zinc-500">
            {search.stage === "processing"
              ? `Processing the other recordings: ${search.done} of ${search.total} (only the first time each is used)`
              : search.stage === "ranking" ? "Ranking everyone…" : "Waiting to start…"}
          </p>
          <div className="mt-1 h-1.5 w-full max-w-xl rounded bg-zinc-200 dark:bg-zinc-800">
            <div className="h-1.5 rounded bg-emerald-600 transition-[width]"
              style={{ width: `${search.total > 0 ? Math.min(100, (search.done / search.total) * 100) : 5}%` }} />
          </div>
        </section>
      )}

      {search?.status === "failed" && (
        <p className="mt-6 text-sm text-red-700 dark:text-red-400">The search failed: {search.error}</p>
      )}

      {search?.status === "complete" && (
        <section className="mt-6">
          <h2 className="font-medium">
            2 · Similar people <span className="text-sm font-normal text-zinc-500">· {MODEL_NAMES[search.model_key] ?? search.model_key}</span>
          </h2>
          <p className="mt-1 text-xs text-zinc-500">
            Best first. A score compares only with scores of the same model. A red card was on screen at the same time
            on the same camera, so it is certainly someone else: it shows how high a wrong match can score.
          </p>
          {search.results.length === 0 ? (
            <p className="mt-2 text-sm text-zinc-500">No one else with a usable crop in these recordings.</p>
          ) : (
            <ol className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {search.results.map((r) => (
                <li key={r.track.track_id}
                  className={`rounded border p-1.5 text-xs ${r.same_time ? "border-red-300 bg-red-50 dark:border-red-800 dark:bg-red-950/40" : "border-zinc-200 dark:border-zinc-700"}`}>
                  <div className="flex h-36 items-center justify-center rounded bg-zinc-100 dark:bg-zinc-900">
                    {r.track.best_crop_uri && (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img src={cropUrl(r.track.best_crop_uri)} alt={`match ${r.rank}`} className="max-h-36 w-auto object-contain" />
                    )}
                  </div>
                  <div className="mt-1 flex items-baseline justify-between font-mono">
                    <span className="font-sans font-medium">#{r.rank}</span>
                    <span title="cosine similarity of one model's barcodes, not a confidence">{r.similarity.toFixed(3)}</span>
                  </div>
                  <div className="mt-0.5 h-1 rounded bg-zinc-200 dark:bg-zinc-800">
                    <div className={`h-1 rounded ${r.same_time ? "bg-red-500" : "bg-emerald-600"}`}
                      style={{ width: `${Math.max(0, r.similarity) * 100}%` }} />
                  </div>
                  <div className="mt-1 text-zinc-500">{r.track.camera_key} · {clock(r.track.started_at)}</div>
                  {r.same_time && <div className="text-red-700 dark:text-red-400">same time: someone else</div>}
                  <button
                    onClick={() => play({ path: r.path, offset: Math.max(0, r.offset - 1),
                      label: `#${r.rank} (${r.track.camera_key} ${clock(r.track.started_at)})` })}
                    className="mt-1 w-full rounded border border-zinc-300 py-0.5 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    ▶ Play
                  </button>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}
    </>
  );
}
