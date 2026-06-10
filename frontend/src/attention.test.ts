import { describe, expect, it } from "vitest";

import { argmax, cellCenter, cropRect, GRID } from "./attention";

describe("attention crop mapping", () => {
  it("argmax finds the peak cell", () => {
    const a = new Array(GRID * GRID).fill(0);
    a[100] = 0.9;
    expect(argmax(a)).toBe(100);
  });

  it("cropRect scales fractions to displayed pixels", () => {
    expect(cropRect({ x: 0.25, y: 0, w: 0.5, h: 1 }, 400, 200)).toEqual({
      left: 100,
      top: 0,
      width: 200,
      height: 200,
    });
  });

  it("the heatmap peak lands at the alpha argmax, inside the crop region", () => {
    const crop = { x: 0.25, y: 0, w: 0.5, h: 1 };
    const [W, H] = [400, 200];
    const a = new Array(GRID * GRID).fill(0);
    a[3 * GRID + 7] = 1; // row 3, col 7
    const center = cellCenter(argmax(a), crop, W, H);
    const rect = cropRect(crop, W, H);
    expect(center.x).toBeGreaterThanOrEqual(rect.left);
    expect(center.x).toBeLessThanOrEqual(rect.left + rect.width);
    expect(center.x).toBeCloseTo(rect.left + ((7 + 0.5) / GRID) * rect.width, 6);
    expect(center.y).toBeCloseTo(((3 + 0.5) / GRID) * H, 6);
  });
});
