/**
 * Placeholder home page. The real screens come in the steps of
 * docs/LEARNING-GUIDE.md:
 *
 *   /search      pick a person crop and "Find similar person" (step 6)
 *   /track/[id]  that person's timeline across cameras (steps 7-8)
 */
const steps = [
  { label: "Frontend", status: "Next.js running", done: true },
  { label: "Backend", status: "Empty Python package, no API yet", done: false },
  { label: "ReID model", status: "Not loaded (OSNet-AIN x1.0, guide step 3)", done: false },
  { label: "Similar-person search", status: "Not built (guide step 6)", done: false },
  { label: "Cross-camera tracking", status: "Not built (guide steps 7-8)", done: false },
];

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-2xl px-4 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">mini-reid</h1>
      <p className="mt-2 text-zinc-600 dark:text-zinc-400">
        A practice project for person re-identification (ReID), built to merge into
        semantic-va later.
      </p>

      <ul className="mt-8 divide-y divide-zinc-200 rounded-lg border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
        {steps.map((s) => (
          <li key={s.label} className="flex items-center justify-between gap-4 px-4 py-3">
            <span className="font-medium">{s.label}</span>
            <span
              className={
                s.done
                  ? "text-sm text-green-700 dark:text-green-400"
                  : "text-sm text-zinc-500"
              }
            >
              {s.status}
            </span>
          </li>
        ))}
      </ul>

      <p className="mt-8 text-sm text-zinc-500">
        Start with <code>docs/ROADMAP.md</code> in the project root.
      </p>
    </main>
  );
}
