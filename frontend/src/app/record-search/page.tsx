import { Suspense } from "react";

import RecordSearchClient from "@/components/RecordSearchClient";

export default function RecordSearchPage() {
  return (
    <Suspense
      fallback={<p className="text-sm text-muted-foreground">Loading search…</p>}
    >
      <div className="flex h-full min-h-0 flex-col">
        <RecordSearchClient />
      </div>
    </Suspense>
  );
}
