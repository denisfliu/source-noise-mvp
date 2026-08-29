"""Live intent bridge (2026-08-29): a background WebSocket hub the pin server uses to
(1) broadcast each replan's proposal — decoded coarse intent path, plain-text sentence,
sigma*, planned chunk — and (2) in GATE mode, block the serve until the human approves,
vetoes+rotates, or overrides sigma. Say-before-do, live, in sim.

Used by serve_gate_pin_joint via SNMVP_INTENT_WS=<port>. Pure stdlib+websockets; runs in
its own thread with an asyncio loop; the serve thread talks to it through thread-safe
primitives only.
"""
import asyncio
import json
import queue
import threading

import numpy as np


class IntentBridge:
    def __init__(self, port):
        self.port = port
        self.mode = "auto"              # "auto" | "gate"
        self.sigma_override = None      # float | None
        self._decision = queue.Queue()  # gate decisions from the UI
        self._clients = set()
        self._loop = None
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    # ---- asyncio side ----
    def _run(self):
        import websockets

        async def handler(ws):
            self._clients.add(ws)
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    if m.get("type") == "mode":
                        self.mode = m["mode"]
                        # a pending gate must not deadlock when the user flips to auto
                        if self.mode == "auto":
                            self._decision.put({"action": "approve"})
                    elif m.get("type") == "sigma":
                        self.sigma_override = None if m["value"] is None else float(m["value"])
                    elif m.get("type") == "decision":
                        self._decision.put(m)
            finally:
                self._clients.discard(ws)

        async def main():
            async with websockets.serve(handler, "127.0.0.1", self.port):
                await asyncio.Future()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(main())

    def _send(self, obj):
        if self._loop is None:
            return
        raw = json.dumps(obj)

        async def go():
            for ws in list(self._clients):
                try:
                    await ws.send(raw)
                except Exception:
                    self._clients.discard(ws)
        asyncio.run_coroutine_threadsafe(go(), self._loop)

    # ---- serve-thread side ----
    def propose(self, payload):
        """Broadcast a proposal. In gate mode, block for the decision; returns it
        ({'action': 'approve'} / {'action': 'rotate', 'deg': +-15} / ...)."""
        payload = dict(payload, type="proposal", gate=(self.mode == "gate"))
        self._send(payload)
        if self.mode != "gate":
            return {"action": "approve"}
        while True:
            try:
                return self._decision.get(timeout=120)
            except queue.Empty:
                self._send({"type": "still_waiting"})

    def executed(self, payload):
        self._send(dict(payload, type="executed"))


def sentence(c, U, astd, cstd):
    """Decode c into a human sentence via the four-horizon vocabulary."""
    path = np.cumsum((U @ c).reshape(50, 32)[:, :4] * astd[:4], axis=0)
    words = []
    for h, label in [(11, "soon"), (45, "over the chunk")]:
        d = path[min(h, 49) - 1]
        parts = []
        if abs(d[0]) > 0.03: parts.append(f"{'fwd' if d[0] > 0 else 'back'} {abs(d[0])*100:.0f}cm")
        if abs(d[1]) > 0.03: parts.append(f"{'left' if d[1] > 0 else 'right'} {abs(d[1])*100:.0f}cm")
        if abs(d[2]) > 0.03: parts.append(f"{'up' if d[2] > 0 else 'down'} {abs(d[2])*100:.0f}cm")
        if abs(d[3]) > 0.10: parts.append(f"turn {'L' if d[3] > 0 else 'R'} {abs(np.degrees(d[3])):.0f}deg")
        if parts:
            words.append(f"{label}: " + ", ".join(parts))
    mag = float(np.linalg.norm(c) / max(np.linalg.norm(cstd), 1e-6))
    return ("; ".join(words) if words else "hold position") + f"  [|c|={mag:.2f} cstd]"
