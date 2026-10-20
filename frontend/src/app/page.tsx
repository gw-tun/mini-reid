import Link from "next/link";
import { Suspense } from "react";

import { Problem } from "@/components/crop";
import { clock, getRecordings } from "@/lib/api";
import { day, searchPage, videoUrl, type Recording, type RecordingGroup } from "@/lib/recordings";

/** The downloaded semantic-va search results. "Search similar" finds a person
 * from one recording in all of them, processing what hasn't been processed. */
export default function Recordings() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">Recordings</h1>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        semantic-va search results, downloaded with <code>scripts/fetch_search_clips.py</code>. Press{" "}
        <b>Search similar</b> on a recording, pick a person in it, and mini-reid looks for them in every recording
        here (YOLO → tracking → ReID barcodes → pgvector, the first time a recording is used).
      </p>
      <Suspense fallback={<p className="mt-8 text-sm text-zinc-500">Loading recordings…</p>}>
        <Groups />
      </Suspense>
    </main>
  );
}

async function Groups() {
  let groups: RecordingGroup[];
  try {
    groups = await getRecordings();
  } catch (error) {
    return <div className="mt-8"><Problem error={error} /></div>;
  }
  if (groups.length === 0) {
    return (
      <p className="mt-8 text-sm text-zinc-600 dark:text-zinc-400">
        No recordings yet. Download some search results first:{" "}
        <code>.venv/bin/python scripts/fetch_search_clips.py &quot;a person in a red top&quot; --source snippet</code>
      </p>
    );
  }
  return (
    <div className="mt-6 space-y-10">
      {groups.map((g) => {
        const processed = g.recordings.filter((r) => r.processed).length;
        return (
          <section key={`${g.at}-${g.text}`}>
            <h2 className="text-lg font-medium">
              {g.text ? <>&ldquo;{g.text}&rdquo;</> : "Other recordings"}{" "}
              <span className="text-sm font-normal text-zinc-500">
                {g.predicate && <>· {g.predicate} </>}
                {g.at && <>· searched {g.at.replace("T", " ")} </>}· {g.recordings.length} recordings · {processed}{" "}
                processed
              </span>
            </h2>
            <ul className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {g.recordings.map((r) => (
                <Card key={r.path} recording={r} />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

function Card({ recording: r }: { recording: Recording }) {
  return (
    <li className="rounded-lg border border-zinc-200 p-2 dark:border-zinc-800">
      <video src={videoUrl(r.path)} controls preload="metadata" className="aspect-video w-full rounded bg-black" />
      <div className="mt-2 flex items-start justify-between gap-2 text-sm">
        <div className="min-w-0">
          <p className="font-medium">
            {r.camera}{" "}
            <span className="font-normal text-zinc-500">
              {r.recorded_from && <>{day(r.recorded_from)} {clock(r.recorded_from)}</>}
              {r.seconds != null && <> · {Math.round(r.seconds)} s</>}
            </span>
          </p>
          <p className="text-xs text-zinc-500">
            {r.rank != null && <>search rank #{r.rank} · </>}
            {r.processed ? (
              <>{r.people ?? 0} people with a usable crop</>
            ) : (
              <>not processed yet</>
            )}
            {r.via === "api" && (
              <span className="text-amber-700 dark:text-amber-400" title="Downloaded at 960 px: small people fail the 96 px crop gate">
                {" "}· 960 px
              </span>
            )}
          </p>
        </div>
        <Link
          href={searchPage(r.path)}
          className="shrink-0 rounded bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-700"
        >
          Search similar
        </Link>
      </div>
    </li>
  );
}
