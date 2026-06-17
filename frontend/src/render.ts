import type { Beam, BlipResult, CaptionResponse, SatResult } from "./api";

type Attr = string | number | boolean | EventListener;

export function h(
  tag: string,
  attrs: Record<string, Attr> = {},
  ...children: (Node | string)[]
): HTMLElement {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value as EventListener);
    } else if (typeof value === "boolean") {
      if (value) node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(value));
    }
  }
  node.append(...children);
  return node;
}

export function clear(root: HTMLElement): void {
  root.replaceChildren();
}

async function asFile(url: string): Promise<File> {
  const blob = await (await fetch(url)).blob();
  return new File([blob], url.split("/").pop() ?? "sample.jpg", {
    type: blob.type || "image/jpeg",
  });
}

export function dropzone(
  onFile: (file: File) => void,
  samples: string[],
): HTMLElement {
  const input = h("input", {
    type: "file",
    accept: "image/*",
    class: "file-input",
  }) as HTMLInputElement;
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) onFile(file);
  });

  const zone = h(
    "div",
    { class: "dropzone" },
    h("p", {}, "drop an image, or "),
    h("label", { class: "browse" }, "browse", input),
  );
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    const file = (e as DragEvent).dataTransfer?.files?.[0];
    if (file) onFile(file);
  });

  const thumbs = samples.map((src) => {
    const img = h("img", { src, class: "sample", alt: "sample" });
    img.addEventListener("click", () => void asFile(src).then(onFile));
    return img;
  });
  const strip = samples.length
    ? h(
        "div",
        { class: "samples" },
        h("span", { class: "samples-label" }, "try to break it:"),
        ...thumbs,
      )
    : h("div", { class: "samples" });

  return h("div", { class: "intake" }, zone, strip);
}

export function statusView(label: string): HTMLElement {
  return h(
    "div",
    { class: "status" },
    h("span", { class: "spinner" }),
    h("span", {}, label),
  );
}

export function wakingPanel(): HTMLElement {
  return h(
    "div",
    { class: "waking" },
    h("span", { class: "pulse" }),
    h("p", { class: "waking-label" }, "system is waking up"),
    h(
      "p",
      { class: "waking-sub" },
      h("span", {}, "cold start takes about a minute · "),
      h("span", { class: "elapsed" }, "0s"),
    ),
  );
}

export function errorView(message: string, onRetry: () => void): HTMLElement {
  return h(
    "div",
    { class: "error" },
    h("p", {}, message),
    h("button", { class: "retry", onClick: onRetry }, "try again"),
  );
}

export function resetButton(onReset: () => void): HTMLElement {
  return h(
    "button",
    { class: "reset", onClick: onReset },
    "caption another image",
  );
}

function wordSpans(words: string[]): HTMLElement[] {
  return words.map((w, i) => h("span", { class: "word", "data-i": i }, w));
}

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

function beamPanel(beams: Beam[]): HTMLElement {
  return h(
    "div",
    { class: "beams" },
    h("h3", {}, "the road not taken"),
    h(
      "ol",
      {},
      ...beams.map((b, i) =>
        h(
          "li",
          { class: i === 0 ? "winner" : "" },
          h("span", { class: "beam-caption" }, b.caption),
          h("span", { class: "beam-score" }, b.score.toFixed(2)),
        ),
      ),
    ),
  );
}

export function satCard(sat: SatResult, imageUrl: string): HTMLElement {
  const figLabel = h(
    "p",
    { class: "fig-label" },
    h("span", {}, "fig.1 — attention"),
    h("span", { class: "fig-word" }, "press play"),
  );
  const figure = h(
    "figure",
    { class: "stage" },
    h("img", { src: imageUrl, class: "preview", alt: "uploaded" }),
    h("canvas", { class: "attn" }),
  );
  const caption = h("p", { class: "caption" });
  wordSpans(sat.words).forEach((span, i) => {
    if (i) caption.append(" ");
    caption.append(span);
  });
  const controls = h(
    "div",
    { class: "controls" },
    h("button", { class: "play", title: "play" }, "▶"),
    h("button", { class: "pause", title: "pause" }, "❚❚"),
    h("button", { class: "step", title: "step" }, "▷❘"),
    h("input", {
      type: "range",
      class: "scrub",
      min: 0,
      max: Math.max(0, sat.words.length - 1),
      value: 0,
    }),
    h("span", { class: "counter" }, `00 / ${pad(sat.words.length)}`),
  );
  return h(
    "div",
    { class: "card glass" },
    h("h2", {}, "capit", h("span", { class: "tag" }, "glass box")),
    figLabel,
    figure,
    caption,
    controls,
    beamPanel(sat.beams),
    h("p", { class: "timing" }, `${sat.decode_ms} ms · beam 3`),
  );
}

export function blipCard(blip: BlipResult): HTMLElement {
  return h(
    "div",
    { class: "card closed" },
    h("h2", {}, "BLIP ", h("span", { class: "tag" }, "closed box")),
    h("p", { class: "caption" }, blip.caption),
    h("p", { class: "timing" }, `${blip.decode_ms} ms`),
  );
}

export function results(
  response: CaptionResponse,
  imageUrl: string,
): HTMLElement {
  const row = h("div", { class: "results" });
  if (response.sat) row.append(satCard(response.sat, imageUrl));
  if (response.blip) row.append(blipCard(response.blip));
  return row;
}
