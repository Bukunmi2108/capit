export interface Crop {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Beam {
  caption: string;
  score: number;
}

export interface SatResult {
  caption: string;
  words: string[];
  attention: number[][];
  crop: Crop;
  beams: Beam[];
  decode_ms: number;
}

export interface BlipResult {
  caption: string;
  decode_ms: number;
}

export interface CaptionResponse {
  sat?: SatResult;
  blip?: BlipResult;
}

export type Model = "sat" | "blip" | "both";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TIMEOUT_MS = 60_000;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function health(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function caption(file: File, model: Model = "both", beam = 3): Promise<CaptionResponse> {
  const form = new FormData();
  form.append("file", file);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${API_URL}/caption?model=${model}&beam=${beam}`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new ApiError(res.status, await res.text().catch(() => res.statusText));
    }
    return (await res.json()) as CaptionResponse;
  } finally {
    clearTimeout(timer);
  }
}
