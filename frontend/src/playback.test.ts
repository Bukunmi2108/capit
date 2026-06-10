import { describe, expect, it } from "vitest";

import { Playback, type Scheduler } from "./playback";

function manualScheduler(): { scheduler: Scheduler; tick: () => void } {
  let pending: (() => void) | null = null;
  return {
    scheduler: {
      set: (cb) => {
        pending = cb;
        return 1;
      },
      clear: () => {
        pending = null;
      },
    },
    tick: () => {
      const cb = pending;
      pending = null;
      cb?.();
    },
  };
}

describe("Playback", () => {
  it("step advances and emits each frame in order", () => {
    const frames: number[] = [];
    const p = new Playback(3, (i) => frames.push(i));
    p.step();
    p.step();
    expect(frames).toEqual([0, 1]);
    expect(p.index).toBe(1);
  });

  it("step stops and pauses at the last word", () => {
    const p = new Playback(2, () => {});
    p.step();
    p.step();
    p.step();
    expect(p.index).toBe(1);
    expect(p.playing).toBe(false);
  });

  it("scrub clamps out-of-range and pauses", () => {
    const frames: number[] = [];
    const p = new Playback(3, (i) => frames.push(i));
    p.scrub(9);
    expect(p.index).toBe(2);
    expect(frames.at(-1)).toBe(2);
    expect(p.playing).toBe(false);
  });

  it("back steps backward, clamped at 0", () => {
    const p = new Playback(3, () => {});
    p.scrub(2);
    p.back();
    expect(p.index).toBe(1);
    p.scrub(0);
    p.back();
    expect(p.index).toBe(0);
  });

  it("play advances via the scheduler then auto-pauses at the end", () => {
    const { scheduler, tick } = manualScheduler();
    const frames: number[] = [];
    const p = new Playback(2, (i) => frames.push(i), 100, scheduler);
    p.play();
    expect(p.playing).toBe(true);
    tick();
    tick();
    tick();
    expect(frames).toEqual([0, 1]);
    expect(p.playing).toBe(false);
  });

  it("pause halts further ticks", () => {
    const { scheduler, tick } = manualScheduler();
    const frames: number[] = [];
    const p = new Playback(5, (i) => frames.push(i), 100, scheduler);
    p.play();
    tick();
    p.pause();
    tick();
    expect(frames).toEqual([0]);
  });
});
