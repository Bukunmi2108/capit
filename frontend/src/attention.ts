import type { Crop } from "./api";

export const GRID = 14;

export interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

/** The center-crop region (fractions of the original) scaled to displayed-image pixels. */
export function cropRect(crop: Crop, imageWidth: number, imageHeight: number): Rect {
  return {
    left: crop.x * imageWidth,
    top: crop.y * imageHeight,
    width: crop.w * imageWidth,
    height: crop.h * imageHeight,
  };
}

/** Pixel center of attention cell `index` (0..195) within the crop region. */
export function cellCenter(index: number, crop: Crop, imageWidth: number, imageHeight: number): { x: number; y: number } {
  const rect = cropRect(crop, imageWidth, imageHeight);
  const row = Math.floor(index / GRID);
  const col = index % GRID;
  return {
    x: rect.left + ((col + 0.5) / GRID) * rect.width,
    y: rect.top + ((row + 0.5) / GRID) * rect.height,
  };
}

/** Paint a 14×14 white-on-transparent heatmap (alpha channel = normalized attention). */
export function paintHeatmap(canvas: HTMLCanvasElement, alpha: number[]): void {
  canvas.width = GRID;
  canvas.height = GRID;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const image = ctx.createImageData(GRID, GRID);
  let max = 0;
  for (const a of alpha) if (a > max) max = a;
  for (let i = 0; i < alpha.length; i++) {
    const v = max > 0 ? alpha[i] / max : 0;
    image.data[i * 4 + 0] = 255;
    image.data[i * 4 + 1] = 255;
    image.data[i * 4 + 2] = 255;
    image.data[i * 4 + 3] = Math.round(v * 255);
  }
  ctx.putImageData(image, 0, 0);
}

/** Lay the heatmap canvas over the crop region of its sibling image (offset-parent positioned). */
export function positionOverlay(canvas: HTMLCanvasElement, image: HTMLImageElement, crop: Crop): void {
  const rect = cropRect(crop, image.clientWidth, image.clientHeight);
  canvas.style.left = `${image.offsetLeft + rect.left}px`;
  canvas.style.top = `${image.offsetTop + rect.top}px`;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}
