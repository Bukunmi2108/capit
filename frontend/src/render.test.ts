import { describe, expect, it } from "vitest";

import type { SatResult } from "./api";
import { blipCard, dropzone, errorView, results, satCard, wakingBanner } from "./render";

const sat: SatResult = {
  caption: "a dog runs",
  words: ["a", "dog", "runs"],
  attention: [new Array(196).fill(0), new Array(196).fill(0), new Array(196).fill(0)],
  crop: { x: 0, y: 0, w: 1, h: 1 },
  beams: [
    { caption: "a dog runs", score: -0.9 },
    { caption: "the dog ran", score: -1.2 },
  ],
  decode_ms: 42,
};

describe("render", () => {
  it("satCard shows word spans, caption, beams (winner first), timing, and an attention canvas", () => {
    const card = satCard(sat, "blob:fake");
    expect(card.querySelectorAll(".word").length).toBe(3);
    expect(card.querySelector(".caption")?.textContent).toContain("dog");
    expect(card.querySelectorAll(".beams li").length).toBe(2);
    expect(card.querySelector(".beams li")?.classList.contains("winner")).toBe(true);
    expect(card.querySelector("canvas.attn")).not.toBeNull();
    expect(card.textContent).toContain("42 ms");
  });

  it("blipCard is deliberately bare: a caption, no word spans or beams", () => {
    const card = blipCard({ caption: "two dogs", decode_ms: 99 });
    expect(card.querySelector(".caption")?.textContent).toBe("two dogs");
    expect(card.querySelectorAll(".word").length).toBe(0);
    expect(card.querySelector(".beams")).toBeNull();
  });

  it("results renders both cards when both models are present", () => {
    const row = results({ sat, blip: { caption: "two dogs", decode_ms: 99 } }, "blob:fake");
    expect(row.querySelector(".card.glass")).not.toBeNull();
    expect(row.querySelector(".card.closed")).not.toBeNull();
  });

  it("wakingBanner and errorView render and the retry button fires", () => {
    expect(wakingBanner().textContent).toContain("waking up");
    let retried = false;
    const view = errorView("could not decode image", () => {
      retried = true;
    });
    view.querySelector("button")?.dispatchEvent(new Event("click"));
    expect(retried).toBe(true);
    expect(view.textContent).toContain("could not decode image");
  });

  it("dropzone calls back when a file input changes", () => {
    const zone = dropzone(() => {}, []);
    expect(zone.querySelector(".dropzone")).not.toBeNull();
  });
});
