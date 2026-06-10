export interface PlaybackSnapshot {
  index: number;
  playing: boolean;
  total: number;
}

export interface Scheduler {
  set(callback: () => void, ms: number): number;
  clear(handle: number): void;
}

const realScheduler: Scheduler = {
  set: (cb, ms) => setTimeout(cb, ms) as unknown as number,
  clear: (handle) => clearTimeout(handle),
};

/**
 * The word-by-word playback engine: a pure state machine over an already-received caption.
 */
export class Playback {
  private idx = -1;
  private active = false;
  private handle: number | null = null;

  constructor(
    private readonly total: number,
    private readonly onFrame: (index: number) => void,
    private readonly intervalMs = 600,
    private readonly scheduler: Scheduler = realScheduler,
  ) {}

  get index(): number {
    return this.idx;
  }

  get playing(): boolean {
    return this.active;
  }

  snapshot(): PlaybackSnapshot {
    return { index: this.idx, playing: this.active, total: this.total };
  }

  private show(i: number): void {
    this.idx = Math.max(0, Math.min(i, this.total - 1));
    this.onFrame(this.idx);
  }

  step(): void {
    if (this.idx < this.total - 1) this.show(this.idx + 1);
    else this.pause();
  }

  back(): void {
    if (this.idx > 0) this.show(this.idx - 1);
  }

  scrub(i: number): void {
    this.pause();
    this.show(i);
  }

  play(): void {
    if (this.active || this.total === 0) return;
    if (this.idx >= this.total - 1) this.idx = -1;
    this.active = true;
    const tick = (): void => {
      if (!this.active) return;
      if (this.idx >= this.total - 1) {
        this.pause();
        return;
      }
      this.show(this.idx + 1);
      this.handle = this.scheduler.set(tick, this.intervalMs);
    };
    this.handle = this.scheduler.set(tick, this.intervalMs);
  }

  pause(): void {
    this.active = false;
    if (this.handle !== null) {
      this.scheduler.clear(this.handle);
      this.handle = null;
    }
  }

  reset(): void {
    this.pause();
    this.idx = -1;
  }
}
