import { promises as fs } from "node:fs";
import path from "node:path";
import type { NextRequest } from "next/server";

const CAMERAS = new Set(["cam01", "cam02", "cam03"]);

export async function GET(request: NextRequest) {
  const camera = request.nextUrl.searchParams.get("camera") ?? "cam01";
  if (!CAMERAS.has(camera)) {
    return Response.json({ error: "unknown camera" }, { status: 400 });
  }
  const directory = path.join(process.cwd(), "..", "data", "recordings", camera);
  let entries: string[];
  try {
    entries = await fs.readdir(directory);
  } catch {
    return Response.json(
      { camera, segments: [] },
      { headers: { "Cache-Control": "no-store" } },
    );
  }
  const segments: { name: string; sizeBytes: number; modifiedAt: string }[] = [];
  for (const name of entries) {
    if (!name.toLowerCase().endsWith(".mp4") || !/^[A-Za-z0-9._-]+$/.test(name)) {
      continue;
    }
    const stats = await fs.stat(path.join(directory, name)).catch(() => null);
    if (stats?.isFile()) {
      segments.push({ name, sizeBytes: stats.size, modifiedAt: stats.mtime.toISOString() });
    }
  }
  segments.sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt));
  return Response.json(
    { camera, segments },
    { headers: { "Cache-Control": "no-store" } },
  );
}
