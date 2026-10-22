import type { NextRequest } from "next/server";

const WHEP_BASE = process.env.MEDIAMTX_WEBRTC_URL ?? "http://127.0.0.1:8889";

export async function POST(
  request: NextRequest,
  ctx: { params: Promise<{ path: string }> },
) {
  const { path } = await ctx.params;
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(path)) {
    return new Response("unknown stream", { status: 404 });
  }
  const offer = await request.text();
  if (!offer.includes("v=0") || !offer.includes("m=video")) {
    return new Response("invalid SDP offer", { status: 400 });
  }
  const upstream = await fetch(`${WHEP_BASE}/${path}/whep`, {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: offer,
    cache: "no-store",
  }).catch(() => null);
  if (upstream === null) {
    return new Response("MediaMTX WebRTC endpoint unreachable", { status: 502 });
  }
  const answer = await upstream.text();
  if (!upstream.ok) {
    return new Response(`MediaMTX WHEP rejected the offer (${upstream.status})`, {
      status: 502,
    });
  }
  return new Response(answer, {
    status: 201,
    headers: { "Content-Type": "application/sdp" },
  });
}
