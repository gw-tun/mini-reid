import { createReadStream, promises as fs } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import type { NextRequest } from "next/server";

const CAMERAS = new Set(["cam01", "cam02", "cam03"]);

export async function GET(
  request: NextRequest,
  ctx: { params: Promise<{ file: string }> },
) {
  const { file } = await ctx.params;
  const camera = request.nextUrl.searchParams.get("camera") ?? "cam01";
  if (!CAMERAS.has(camera) || !/^[A-Za-z0-9._-]{1,128}$/.test(file) || file.includes("..")) {
    return new Response("not found", { status: 404 });
  }
  const filePath = path.join(process.cwd(), "..", "data", "recordings", camera, file);
  const stats = await fs.stat(filePath).catch(() => null);
  if (!stats?.isFile()) {
    return new Response("not found", { status: 404 });
  }
  const headers: Record<string, string> = {
    "Content-Type": "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-store",
  };
  const range = request.headers.get("range");
  const match = range ? /^bytes=(\d*)-(\d*)$/.exec(range.trim()) : null;
  if (range && !match) {
    return new Response("unsupported range", { status: 416, headers });
  }
  let start = 0;
  let end = stats.size - 1;
  if (match) {
    if (match[1] === "" && match[2] === "") {
      return new Response("invalid range", { status: 416, headers });
    }
    if (match[1] === "") {
      start = Math.max(stats.size - Number(match[2]), 0);
    } else {
      start = Number(match[1]);
    }
    if (match[2] !== "") {
      end = Math.min(Number(match[2]), stats.size - 1);
    }
    if (start > end || start >= stats.size) {
      return new Response("range not satisfiable", {
        status: 416,
        headers: { ...headers, "Content-Range": `bytes */${stats.size}` },
      });
    }
    const stream = Readable.toWeb(
      createReadStream(filePath, { start, end }),
    ) as ReadableStream<Uint8Array>;
    return new Response(stream, {
      status: 206,
      headers: {
        ...headers,
        "Content-Range": `bytes ${start}-${end}/${stats.size}`,
        "Content-Length": String(end - start + 1),
      },
    });
  }
  const stream = Readable.toWeb(createReadStream(filePath)) as ReadableStream<Uint8Array>;
  return new Response(stream, {
    status: 200,
    headers: { ...headers, "Content-Length": String(stats.size) },
  });
}
