import { ApiError, cropUrl } from "@/lib/api";

/** A person crop from the API. Plain <img>: crops are small JPEGs the API
 * already serves at their real size, so Next's image optimizer adds nothing. */
export function Crop({ uri, alt, className = "h-40" }: { uri: string | null; alt: string; className?: string }) {
  if (!uri) {
    return (
      <div className={`${className} flex items-center justify-center rounded bg-zinc-100 px-2 text-center text-xs text-zinc-500 dark:bg-zinc-900`}>
        no usable crop
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={cropUrl(uri)} alt={alt} className={`${className} w-auto rounded bg-zinc-100 object-contain dark:bg-zinc-900`} />
  );
}

/** What to show when the API answered with an error, or isn't running. */
export function Problem({ error }: { error: unknown }) {
  const message = error instanceof ApiError ? error.message : String(error);
  return (
    <p className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
      {message}
    </p>
  );
}
