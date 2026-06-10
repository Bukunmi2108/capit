import "./style.css";

import { ApiError, caption, type CaptionResponse, health, type SatResult } from "./api";
import { paintHeatmap, positionOverlay } from "./attention";
import { Playback } from "./playback";
import { clear, dropzone, errorView, resetButton, results, statusView, wakingBanner } from "./render";

const SAMPLES = ["/samples/sample-1.jpg", "/samples/sample-2.jpg", "/samples/sample-3.jpg"];

const root = document.getElementById("app");
if (!root) throw new Error("missing #app element");
const app = root;

let requestSeq = 0;

function errorMessage(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 413) return "that image is too large (max 8 MB).";
    if (e.status === 422) return "that doesn't look like an image we can read.";
    return `the server returned an error (${e.status}).`;
  }
  if (e instanceof DOMException && e.name === "AbortError") return "the request timed out — the model may be waking up.";
  return "could not reach the captioner. is the backend running?";
}

function showIdle(waking: boolean): void {
  clear(app);
  if (waking) app.append(wakingBanner());
  app.append(dropzone(onFile, SAMPLES));
}

async function onFile(file: File): Promise<void> {
  const seq = ++requestSeq;
  clear(app);
  app.append(statusView("captioning…"));
  try {
    const response = await caption(file, "both", 3);
    if (seq !== requestSeq) return; // a newer upload superseded this one
    showResults(file, response);
  } catch (e) {
    if (seq !== requestSeq) return;
    clear(app);
    app.append(errorView(errorMessage(e), () => showIdle(false)));
  }
}

function showResults(file: File, response: CaptionResponse): void {
  clear(app);
  const url = URL.createObjectURL(file);
  const row = results(response, url);
  app.append(row, resetButton(() => showIdle(false)));
  if (response.sat) wireGlassBox(row, response.sat);
}

function wireGlassBox(row: HTMLElement, sat: SatResult): void {
  const card = row.querySelector(".card.glass");
  if (!card) return;
  const image = card.querySelector("img.preview") as HTMLImageElement;
  const canvas = card.querySelector("canvas.attn") as HTMLCanvasElement;
  const words = Array.from(card.querySelectorAll<HTMLElement>(".word"));
  const scrub = card.querySelector(".scrub") as HTMLInputElement;
  const counter = card.querySelector<HTMLElement>(".counter");
  const figWord = card.querySelector<HTMLElement>(".fig-word");
  const pad = (n: number): string => String(n).padStart(2, "0");

  const frame = (i: number): void => {
    words.forEach((w, j) => w.classList.toggle("active", j === i));
    scrub.value = String(i);
    if (counter) counter.textContent = `${pad(i + 1)} / ${pad(words.length)}`;
    if (figWord) figWord.textContent = `“${sat.words[i]}”`;
    paintHeatmap(canvas, sat.attention[i]);
    positionOverlay(canvas, image, sat.crop);
    canvas.classList.add("show");
  };

  const playback = new Playback(sat.words.length, frame);
  card.querySelector(".play")?.addEventListener("click", () => playback.play());
  card.querySelector(".pause")?.addEventListener("click", () => playback.pause());
  card.querySelector(".step")?.addEventListener("click", () => playback.step());
  scrub.addEventListener("input", () => playback.scrub(Number(scrub.value)));
  words.forEach((w, i) =>
    w.addEventListener("mouseenter", () => {
      playback.pause();
      frame(i);
    }),
  );
  image.addEventListener("load", () => positionOverlay(canvas, image, sat.crop));
  window.addEventListener("resize", () => positionOverlay(canvas, image, sat.crop));
}

async function init(): Promise<void> {
  showIdle(!(await health()));
}

void init();
