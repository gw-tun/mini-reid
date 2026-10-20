import Link from "next/link";
import { Suspense } from "react";

import { SimilarSearch } from "@/components/similar-search";

/** "Search similar" for one downloaded recording: pick a person in it, find them in all recordings. */
export default function RecordingSearchPage({ params }: PageProps<"/recordings/[...path]">) {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8">
      <Link href="/" className="text-sm text-zinc-500 hover:underline">
        ← All recordings
      </Link>
      <Suspense fallback={<p className="mt-8 text-sm text-zinc-500">Loading…</p>}>
        {params.then(({ path }) => (
          <SimilarSearch path={path.map(decodeURIComponent).join("/")} />
        ))}
      </Suspense>
    </main>
  );
}
