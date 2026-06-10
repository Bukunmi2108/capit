import { afterEach, describe, expect, it, vi } from "vitest";

import { caption, health } from "./api";

afterEach(() => vi.restoreAllMocks());

function mockFetch(impl: () => Promise<Response>): void {
  vi.stubGlobal("fetch", vi.fn(impl));
}

const file = new File([new Uint8Array([1, 2, 3])], "x.png", { type: "image/png" });

describe("health", () => {
  it("is true when /health responds ok", async () => {
    mockFetch(async () => new Response("{}", { status: 200 }));
    expect(await health()).toBe(true);
  });

  it("is false when the network rejects (waking/down)", async () => {
    mockFetch(async () => {
      throw new Error("offline");
    });
    expect(await health()).toBe(false);
  });
});

describe("caption", () => {
  it("returns the parsed response on 200", async () => {
    const body = {
      sat: { caption: "a dog", words: ["a", "dog"], attention: [[0], [0]], crop: { x: 0, y: 0, w: 1, h: 1 }, beams: [], decode_ms: 1 },
      blip: { caption: "a dog", decode_ms: 2 },
    };
    mockFetch(async () => new Response(JSON.stringify(body), { status: 200 }));
    const r = await caption(file, "both", 3);
    expect(r.sat?.caption).toBe("a dog");
    expect(r.blip?.caption).toBe("a dog");
  });

  it("throws ApiError carrying the status on 422", async () => {
    mockFetch(async () => new Response("could not decode image", { status: 422 }));
    await expect(caption(file)).rejects.toMatchObject({ status: 422, name: "ApiError" });
  });

  it("propagates a network error", async () => {
    mockFetch(async () => {
      throw new Error("network down");
    });
    await expect(caption(file)).rejects.toThrow("network down");
  });
});
