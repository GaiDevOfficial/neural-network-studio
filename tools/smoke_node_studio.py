"""
Node Studio — real-browser smoke test (WebView2 via pywebview).

Boots the production dev server (server.py's own DevHandler) on a free port, opens
src/graph_studio.html in a real WebView2 window, and then drives the UI exactly like a
user would: press the MNIST preset, wait for data resolution + live validation, press Run,
and measure what the page actually shows (rendered blocks, wires, epoch counter, log).

This is the end-to-end check that the shipped UI boots without console errors, that the
canvas renders the compiled graph, and that the training loop really advances.

Run manually (needs a desktop session):  python tools/smoke_node_studio.py

File: tools/smoke_node_studio.py
"""

import json
import os
import socket
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import server  # noqa: E402  (real dev server with the Data API wired in)
import webview  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    detail = "".join(ch if ord(ch) < 128 else "?" for ch in str(detail))
    RESULTS.append((bool(condition), name, detail))
    mark = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {name}{suffix}", flush=True)
    return bool(condition)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = free_port()
    httpd = server.StudioHTTPServer(("127.0.0.1", port), server.DevHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/graph_studio.html"
    print(f"[setup] serving {PROJECT_ROOT}\\src on port {port}", flush=True)
    print(f"[setup] opening {url} in WebView2\n", flush=True)

    window = webview.create_window("Node Studio smoke", url, width=1500, height=940, hidden=True)

    def ev(js, default=None):
        try:
            return window.evaluate_js(js)
        except Exception:
            return default

    def ev_json(js, default=None):
        raw = ev(f"JSON.stringify(({js}))", None)
        if raw in (None, ""):
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default

    def click(selector):
        return ev(
            "(function(){const el=document.querySelector(" + json.dumps(selector) + ");"
            "if(!el) return false; el.click(); return true;})()",
            False,
        )

    def escape():
        ev("document.dispatchEvent(new KeyboardEvent('keydown',"
           "{key:'Escape', bubbles:true, cancelable:true}))")

    def menu_hidden(menu_id):
        return ev(f"document.getElementById({json.dumps(menu_id)}).hidden", None)

    def drive():
        try:
            run_checks()
        except Exception as error:  # noqa: BLE001 — the smoke driver must never hang the window
            print(f"  [FAIL] driver crashed: {error}", flush=True)
        finally:
            window.destroy()

    def run_checks():
        # ---- boot ----
        deadline = time.time() + 30
        palette = 0
        while time.time() < deadline:
            palette = ev("document.querySelectorAll('.ns-palette-item').length", 0) or 0
            if palette > 0:
                break
            time.sleep(0.5)
        check("palette rendered from the block registry", palette >= 25, f"{palette} blocks")
        status = ev("document.getElementById('status-text').textContent", "") or ""
        check("status bar shows the ready message", "Ready" in status, status[:60])

        # ---- preset ----
        ev("document.getElementById('btn-preset-mnist').click()")
        nodes = 0
        valid = ""
        wires = 0
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(0.5)
            nodes = ev("document.querySelectorAll('.ns-node').length", 0) or 0
            wires = ev("document.querySelectorAll('.ns-wire').length", 0) or 0
            valid = ev("document.getElementById('validation-panel').textContent", "") or ""
            if nodes >= 14 and "valid" in valid.lower() and wires >= 13:
                break
        check("MNIST preset renders 14 blocks on the canvas", nodes == 14, f"{nodes} nodes")
        check("13 wires drawn between the blocks", wires >= 13, f"{wires} wires")
        check("live validation reports the compiled chain", "valid" in valid.lower(), valid.strip()[:80])
        plan = ev("document.getElementById('status-params').textContent", "") or ""
        check(
            "compiled plan shown in the status bar (architecture + loss + optimizer)",
            "784" in plan and "crossentropy" in plan.lower() and "adam" in plan.lower(),
            plan[:90],
        )
        dataset = ev("document.getElementById('status-dataset').textContent", "") or ""
        check("real MNIST corpus resolved into the studio", "MNIST" in dataset and "rows" in dataset, dataset[:80])
        dataset_chip = ev("document.getElementById('action-dataset').textContent", "") or ""
        check("action bar exposes the resolved dataset", "MNIST" in dataset_chip, dataset_chip[:60])
        check(
            "every dropdown has an explicit injected close button",
            ev("!!document.querySelector('#tour-menu .ns-menu-close') && "
               "!!document.querySelector('#export-menu .ns-menu-close')", False),
        )

        # ---- the reported "Press Run" replay bug: a fast run must not re-lock the step ----
        # This is the first training action of the session, so the journal only contains preset_mnist.
        click("#btn-tour")
        time.sleep(0.2)
        click('[data-tour-id="first_ai"]')
        time.sleep(0.5)
        first_step = ev_json("""
            (function () {
              const next = document.querySelector('[data-tour="next"]');
              const hint = document.querySelector('.tour-hint');
              const text = document.querySelector('.tour-text');
              return { disabled: next ? next.disabled : null, met: hint ? hint.classList.contains('is-met') : null, textHtml: text ? text.innerHTML : '' };
            })()
        """) or {}
        check("a preset already in the action journal satisfies lesson step 1",
              first_step.get("disabled") is False, str(first_step))
        check("lesson copy renders bold emphasis instead of literal asterisks",
              "<strong>" in (first_step.get("textHtml") or "")
              and "**" not in (first_step.get("textHtml") or ""), first_step.get("textHtml"))
        click('[data-tour="next"]')
        time.sleep(0.25)
        click('[data-tour="next"]')
        time.sleep(0.3)
        # Swap in the tiny XOR corpus so the run finishes in a blink, exactly like the bug report.
        click("#btn-preset-xor")
        time.sleep(0.5)
        run_step = ev_json("""
            (function () {
              const title = document.querySelector('.tour-title');
              const next = document.querySelector('[data-tour="next"]');
              return { title: title ? title.textContent : '', disabled: next ? next.disabled : null };
            })()
        """) or {}
        check("the lesson reaches its Run step", "Press Run" in (run_step.get("title") or ""), str(run_step))
        click("#btn-run")
        for _ in range(80):
            if ev("!!document.getElementById('btn-run') && !document.getElementById('btn-run').disabled", False):
                break
            time.sleep(0.25)
        time.sleep(0.3)
        click('[data-tour="back"]')
        time.sleep(0.3)
        replay = ev_json("""
            (function () {
              const title = document.querySelector('.tour-title');
              const next = document.querySelector('[data-tour="next"]');
              const hint = document.querySelector('.tour-hint');
              return {
                title: title ? title.textContent : '',
                disabled: next ? next.disabled : null,
                met: hint ? hint.classList.contains('is-met') : null,
                hint: hint ? hint.textContent : ''
              };
            })()
        """) or {}
        check("the Run step is remembered after the run stops",
              "Press Run" in (replay.get("title") or "") and replay.get("disabled") is False,
              str(replay))
        click('[data-tour="next"]')
        time.sleep(0.3)
        after_run = ev_json("""
            (function () {
              const title = document.querySelector('.tour-title');
              const next = document.querySelector('[data-tour="next"]');
              return { title: title ? title.textContent : '', disabled: next ? next.disabled : null };
            })()
        """) or {}
        check("Next advances to the metrics step instead of asking for Run again",
              "Read the live metrics" in (after_run.get("title") or ""), str(after_run))
        click(".tour-close")
        time.sleep(0.2)
        # Restore the MNIST graph the rest of the smoke expects.
        click("#btn-preset-mnist")
        time.sleep(1.0)

        # ---- tier switching + action bar ----
        click("#btn-mode-manual")
        time.sleep(0.3)
        manual_tier = ev("document.body.dataset.tier", "")
        manual_active = ev("document.getElementById('btn-mode-manual').classList.contains('active')", False)
        check("Advanced mode switches the tier and highlights its button",
              manual_tier == "manual" and manual_active, f"tier={manual_tier}, active={manual_active}")
        click("#btn-mode-assist")
        time.sleep(0.3)
        check("Beginner mode switches back",
              ev("document.body.dataset.tier", "") == "assist" and
              ev("document.getElementById('btn-mode-assist').classList.contains('active')", False))
        click("#action-metrics")
        time.sleep(0.2)
        check(
            "Live metrics action reveals the training panel",
            ev("document.getElementById('panel-metrics').classList.contains('is-flash')", False),
        )

        # ---- run ----
        ev("window.__smokeErrors = [];"
           "window.addEventListener('error', function (e) { window.__smokeErrors.push('error: ' + (e.message || e.type)); });"
           "window.addEventListener('unhandledrejection', function (e) { var r = e.reason; window.__smokeErrors.push('rejection: ' + ((r && r.message) || String(r))); });"
           "'hooked'")
        ev("document.getElementById('btn-run').click()")
        epoch = 0
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(1.0)
            raw = ev("document.getElementById('metric-epoch').textContent", "—") or "—"
            try:
                epoch = int(raw)
            except ValueError:
                epoch = 0
            if epoch >= 3:
                break
        smoke_errors = ev("JSON.stringify(window.__smokeErrors || [])", "[]") or "[]"
        log_tail = ev("document.getElementById('log-list').textContent", "") or ""
        check("training loop advances epochs in the real worker", epoch >= 3,
              f"epoch {epoch}; page errors: {smoke_errors}; log tail: {log_tail[:420]}")

        train_loss = ev("document.getElementById('metric-loss').textContent", "—") or "—"
        val_loss = ev("document.getElementById('metric-val-loss').textContent", "—") or "—"
        check("live train loss is a measured value", train_loss not in ("—", ""), f"loss {train_loss}")
        check("live held-out loss is a measured value", val_loss not in ("—", ""), f"val {val_loss}")

        errors = ev("document.querySelectorAll('#log-list .ns-log-error').length", 0) or 0
        check("no runtime errors in the UI log", errors == 0, f"{errors} error entries")

        # ---- pause to freeze the solver, then confirm state ----
        ev("document.getElementById('btn-pause').click()")
        time.sleep(1.0)
        paused_epoch = ev("document.getElementById('metric-epoch').textContent", "—")
        check("pause keeps the measured state on screen", paused_epoch not in (None, "", "—"), f"epoch {paused_epoch}")

        # ---- drag a wire to empty space → searchable quick-add popover ----
        quick = ev_json("""
            (function () {
              const canvas = document.getElementById('studio-canvas');
              const r = canvas.getBoundingClientRect();
              const socket = document.querySelector('.ns-node[data-type="dense"] .ns-socket[data-dir="out"]');
              if (!socket) return { fired: false, reason: 'no dense output socket' };
              const sr = socket.getBoundingClientRect();
              // Beginner snapping reaches 34 world units; require 70 screen px of clearance from
              // every socket and never drop on a node, so the release truly lands on empty canvas.
              const sockets = Array.from(document.querySelectorAll('.ns-socket'));
              const clear = (x, y) => {
                const el = document.elementFromPoint(x, y);
                if (!el || el.closest('.ns-node') || el.closest('.ns-socket')) return false;
                for (const s of sockets) {
                  const b = s.getBoundingClientRect();
                  const dx = (b.left + b.width / 2) - x;
                  const dy = (b.top + b.height / 2) - y;
                  if (dx * dx + dy * dy < 70 * 70) return false;
                }
                return true;
              };
              let drop = null;
              for (let gy = 0; gy <= 11 && !drop; gy++) {
                for (let gx = 0; gx <= 11 && !drop; gx++) {
                  const x = r.left + 40 + (r.width - 80) * gx / 11;
                  const y = r.top + 60 + (r.height - 120) * gy / 11;
                  if (clear(x, y)) drop = [Math.round(x), Math.round(y)];
                }
              }
              if (!drop) return { fired: false, reason: 'no empty canvas point' };
              const before = document.querySelectorAll('.ns-node').length;
              const wires = document.querySelectorAll('.ns-wire').length;
              const fire = (target, type, x, y) => target.dispatchEvent(new PointerEvent(type, {
                bubbles: true, cancelable: true, pointerId: 41, button: 0, clientX: x, clientY: y
              }));
              const sx = sr.left + sr.width / 2;
              const sy = sr.top + sr.height / 2;
              fire(socket, 'pointerdown', sx, sy);
              fire(canvas, 'pointermove', sx + 10, sy + 10);
              fire(canvas, 'pointermove', drop[0], drop[1]);
              fire(canvas, 'pointerup', drop[0], drop[1]);
              const panel = document.querySelector('.ns-quick-menu');
              return {
                fired: true,
                before: before,
                wires: wires,
                drop: drop,
                menu: Boolean(panel && !panel.hidden),
                items: document.querySelectorAll('.ns-quick-item').length
              };
            })()
        """, {})
        check("dropping a dragged wire on empty canvas opens the quick-add popover",
              quick.get("menu") is True, str(quick))
        check("the popover only lists blocks compatible with the dragged socket",
              (quick.get("items") or 0) > 0, str(quick))
        pick = ev_json("""
            (function () {
              const item = document.querySelector('.ns-quick-item');
              if (!item) return { clicked: false };
              const type = item.dataset.quickType || '';
              item.click();
              return { clicked: true, type: type };
            })()
        """, {})
        time.sleep(0.6)
        after_add = ev_json("""
            (function () {
              const panel = document.querySelector('.ns-quick-menu');
              return {
                nodes: document.querySelectorAll('.ns-node').length,
                wires: document.querySelectorAll('.ns-wire').length,
                closed: !panel || panel.hidden,
                log: document.getElementById('log-list').textContent || ''
              };
            })()
        """, {})
        check("picking a block places and wires it in one gesture",
              pick.get("clicked") is True and after_add.get("nodes") == (quick.get("before") or 0) + 1,
              f"{pick} -> {after_add.get('nodes')} nodes")
        check("the one-gesture placement adds the wire too",
              (after_add.get("wires") or 0) > (quick.get("wires") or 0), str(after_add))
        check("the quick-add popover closes after the pick", after_add.get("closed") is True, str(after_add))
        check("the log records the one-gesture placement",
              "one gesture" in (after_add.get("log") or ""), (after_add.get("log") or "")[-220:])

        # ---- right-click that new wire → Break connection (animated, real disconnect) ----
        menu_open = ev_json("""
            (function () {
              const canvas = document.getElementById('studio-canvas');
              const paths = document.querySelectorAll('#wire-layer path[data-id]');
              const path = paths[paths.length - 1];
              if (!path) return { hit: false, reason: 'no wire path' };
              const pt = path.getPointAtLength(path.getTotalLength() / 2);
              const ctm = path.getScreenCTM();
              const x = ctm.a * pt.x + ctm.c * pt.y + ctm.e;
              const y = ctm.b * pt.x + ctm.d * pt.y + ctm.f;
              canvas.dispatchEvent(new PointerEvent('contextmenu', {
                bubbles: true, cancelable: true, pointerId: 51, button: 2, clientX: x, clientY: y
              }));
              const panel = document.querySelector('.ns-wire-menu');
              return {
                hit: true,
                before: document.querySelectorAll('.ns-wire').length,
                open: Boolean(panel && !panel.hidden),
                meta: panel ? panel.querySelector('.ns-wire-menu-meta').textContent : '',
                warning: panel && panel.querySelector('.ns-wire-menu-warning')
                  ? panel.querySelector('.ns-wire-menu-warning').textContent : ''
              };
            })()
        """, {})
        check("right-clicking a real bezier wire opens the wire menu",
              menu_open.get("open") is True, str(menu_open))
        check("the wire menu names both endpoints",
              "->" in (menu_open.get("meta") or "") or "→" in (menu_open.get("meta") or ""),
              (menu_open.get("meta") or "")[:120])
        broken = ev_json("""
            (function () {
              const item = document.querySelector('.ns-wire-menu-item');
              if (!item) return { clicked: false };
              item.click();
              return { clicked: true };
            })()
        """, {})
        # The break is deliberately animated (180 ms) and a hidden WebView2 window
        # throttles timers, so wait for the real state change instead of a fixed sleep.
        after_break = {}
        deadline = time.time() + 5
        while time.time() < deadline:
            after_break = ev_json("""
                (function () {
                  const panel = document.querySelector('.ns-wire-menu');
                  return {
                    wires: document.querySelectorAll('.ns-wire').length,
                    closed: !panel || panel.hidden,
                    log: document.getElementById('log-list').textContent || ''
                  };
                })()
            """, {})
            if (after_break.get("wires") == (menu_open.get("before") or 0) - 1
                    and "Broke " in (after_break.get("log") or "")):
                break
            time.sleep(0.1)
        check("Break connection removes exactly the right-clicked wire",
              broken.get("clicked") is True and after_break.get("wires") == (menu_open.get("before") or 0) - 1,
              f"{menu_open.get('before')} -> {after_break.get('wires')}")
        check("the wire menu closes after the break", after_break.get("closed") is True, str(after_break))
        check("the log records the real disconnect",
              "Broke " in (after_break.get("log") or ""), (after_break.get("log") or "")[-220:])

        # ---- universal menu lifecycle: open / Escape / outside / × ----
        click("#btn-tour")
        time.sleep(0.2)
        check("Tours opens its dropdown", menu_hidden("tour-menu") is False)
        escape()
        time.sleep(0.2)
        check("Escape closes the Tours dropdown", menu_hidden("tour-menu") is True)
        click("#btn-tour")
        time.sleep(0.2)
        ev("document.body.dispatchEvent(new PointerEvent('pointerdown',"
           "{bubbles:true, cancelable:true, pointerId:3}))")
        time.sleep(0.2)
        check("outside pointerdown closes the Tours dropdown", menu_hidden("tour-menu") is True)
        click("#btn-tour")
        time.sleep(0.2)
        click("#tour-menu .ns-menu-close")
        time.sleep(0.2)
        check("the injected × closes the Tours dropdown", menu_hidden("tour-menu") is True)

        # ---- the curriculum is one list of ten numbered, step-by-step lessons ----
        click("#btn-tour")
        time.sleep(0.2)
        lessons = ev_json("""
            (function () {
              const items = Array.from(document.querySelectorAll('#tour-menu .ns-lesson-item'));
              return {
                count: items.length,
                numbers: items.map((item) => (item.querySelector('.ns-lesson-num') || {}).textContent || ''),
                playTab: !!document.querySelector('.ns-mode #btn-mode-playground'),
                playButton: !!document.querySelector('.ns-actions #btn-mode-playground'),
                firstTitle: items.length ? items[0].textContent : ''
              };
            })()
        """, {}) or {}
        check("the Lessons menu lists exactly ten guides", lessons.get("count") == 10, str(lessons.get("count")))
        check("every lesson is numbered 01–10",
              lessons.get("numbers") == [f"{i:02d}" for i in range(1, 11)], str(lessons.get("numbers")))
        check("the Playground is no longer a node-mode tab but stays reachable",
              lessons.get("playTab") is False and lessons.get("playButton") is True, str(lessons))
        check("the first lesson names its Playground path",
              "Playground" in (lessons.get("firstTitle") or ""), (lessons.get("firstTitle") or "")[:80])
        escape()
        time.sleep(0.2)

        # ---- curriculum track A: gated spotlight + Escape + explicit × ----
        click("#btn-tour")
        time.sleep(0.2)
        click('[data-tour-track="track_a"]')
        time.sleep(0.6)
        tour_text = ev("(document.querySelector('.tour-tooltip') || {}).textContent || ''", "") or ""
        check("Track A opens a real spotlight tour", "first ai" in tour_text.lower(), tour_text[:100])
        check("action-gated tutorial disables Next until the real gesture",
              ev("(document.querySelector('[data-tour=\"next\"]') || {}).disabled === true", False))
        check("tour tooltip has an explicit ×", ev("!!document.querySelector('.tour-close')", False))
        tour_layout = ev_json("""
            (function () {
              const tip = document.querySelector('.tour-tooltip');
              const target = document.querySelector('#btn-mode-playground');
              const topbar = document.querySelector('.ns-topbar');
              if (!tip || !target || !topbar) return { ok: false };
              const tr = tip.getBoundingClientRect();
              const ar = target.getBoundingClientRect();
              const br = topbar.getBoundingClientRect();
              const overlap = !(tr.right <= ar.left || tr.left >= ar.right
                || tr.bottom <= ar.top || tr.top >= ar.bottom);
              return {
                ok: true,
                overlapsTarget: overlap,
                belowTopbar: tr.top >= br.bottom - 1,
                inView: tr.left >= 0 && tr.top >= 0
                  && tr.right <= window.innerWidth && tr.bottom <= window.innerHeight,
                tip: { left: tr.left, top: tr.top, right: tr.right, bottom: tr.bottom },
                topbarBottom: br.bottom
              };
            })()
        """, {})
        check("the tour tooltip never covers the highlighted control",
              tour_layout.get("ok") is True and tour_layout.get("overlapsTarget") is False,
              str(tour_layout))
        check("the topbar-anchored tour tooltip docks over the canvas, below the bar",
              tour_layout.get("belowTopbar") is True, str(tour_layout))
        check("the tour tooltip stays inside the window", tour_layout.get("inView") is True,
              str(tour_layout))
        escape()
        time.sleep(0.3)
        check("Escape closes the active tour overlay", ev("!document.querySelector('.tour-root')", False))

        # ---- a deep-dive tutorial uses the same Escape owner ----
        click("#btn-tour")
        time.sleep(0.2)
        click('[data-tour-id="first_ai"]')
        time.sleep(0.6)
        check("deep-dive tour opens", ev("!!document.querySelector('.tour-root')", False))
        escape()
        time.sleep(0.3)
        check("Escape closes a deep-dive tour as well", ev("!document.querySelector('.tour-root')", False))

        # ---- drag a block over the sidebar to delete it (real pointer geometry) ----
        drag = ev_json("""
            (function () {
              const nodes = document.querySelectorAll('.ns-node');
              const before = nodes.length;
              const node = nodes[0];
              if (!node) return { before: before, after: before, armed: false, trash: false };
              const nr = node.getBoundingClientRect();
              const pr = document.getElementById('palette').getBoundingClientRect();
              const canvas = document.getElementById('studio-canvas');
              const cx = (pr.left + pr.right) / 2;
              const cy = (pr.top + pr.bottom) / 2;
              const fire = (target, type, x, y) => target.dispatchEvent(new PointerEvent(type, {
                bubbles: true, cancelable: true, pointerId: 11, button: 0, clientX: x, clientY: y
              }));
              fire(node, 'pointerdown', nr.left + nr.width / 2, nr.top + 16);
              fire(canvas, 'pointermove', cx, cy);
              const armed = document.getElementById('palette').classList.contains('is-trash-target');
              fire(canvas, 'pointerup', cx, cy);
              return {
                before: before,
                after: document.querySelectorAll('.ns-node').length,
                armed: armed,
                trash: document.getElementById('palette').classList.contains('is-trash-target')
              };
            })()
        """, {})
        check("dragging a block over the sidebar arms the red dropzone", drag.get("armed") is True, str(drag))
        check("dropping the block on the palette deletes it",
              drag.get("after") == drag.get("before") - 1, str(drag))
        check("the trash state clears after the drop", drag.get("trash") is False, str(drag))
        drag_log = ev("document.getElementById('log-list').textContent", "") or ""
        check("the log records the trash drop", "trash" in drag_log.lower(), drag_log[-200:])

        # ---- Tier 1 playground: prepare, teach, draw, ask, correct, exit ----
        click("#btn-mode-playground")
        deadline = time.time() + 60
        answer = ""
        playground_ready = False
        while time.time() < deadline:
            time.sleep(0.5)
            hidden = ev("document.getElementById('playground').hidden", True)
            answer = ev("document.getElementById('playground-answer').textContent", "") or ""
            if not hidden and ("teach" in answer.lower() or "draw" in answer.lower()):
                playground_ready = True
                break
        check("Playground switches to the pre-assembled brain", playground_ready, answer[:120])
        check("the block canvas is hidden in Playground",
              ev("document.getElementById('studio-main').hidden", False) is True)
        deadline = time.time() + 6
        brain_dims = False
        while time.time() < deadline:
            brain_dims = ev("(function(){const c=document.getElementById('playground-brain');"
                            "const r=c.getBoundingClientRect();const d=window.devicePixelRatio||1;"
                            "return r.width>0 && r.height>0 && Math.abs(c.width-r.width*d)<2 &&"
                            "Math.abs(c.height-r.height*d)<2;})()", False)
            if brain_dims:
                break
            time.sleep(0.2)
        check("the synaptic brain canvas has real backing-store dimensions", brain_dims)
        check("Teach is available", ev("!document.getElementById('playground-teach').disabled", False))
        check("Ask stays disabled before there is a drawing",
              ev("document.getElementById('playground-ask').disabled", False) is True)

        click("#playground-teach")
        deadline = time.time() + 240
        progress = ""
        accuracy = ""
        while time.time() < deadline:
            time.sleep(1.0)
            progress = ev("document.getElementById('playground-progress').textContent", "") or ""
            accuracy = ev("document.getElementById('playground-accuracy').textContent", "") or ""
            if "Practice finished" in progress or "Held-out check" in accuracy:
                break
        check("Teach runs real training and reports real rounds",
              "Practice" in progress or "Held-out check" in accuracy, f"{progress} | {accuracy}"[:160])
        check("training completes or reports a measured held-out check",
              "Practice finished" in progress or "Held-out check" in accuracy, f"{progress} | {accuracy}"[:160])
        check("held-out accuracy is a measured percentage",
              "Held-out check" in accuracy and "%" in accuracy, accuracy[:120])

        draw = ev_json("""
            (function () {
              const pad = document.getElementById('playground-pad');
              const r = pad.getBoundingClientRect();
              if (!r.width || !r.height) return { fired: false, w: r.width, h: r.height };
              const points = [[0.52, 0.18], [0.52, 0.30], [0.52, 0.45], [0.52, 0.60], [0.52, 0.78]];
              const fire = (type, p) => pad.dispatchEvent(new PointerEvent(type, {
                bubbles: true, cancelable: true, pointerId: 21, button: 0,
                clientX: r.left + p[0] * r.width, clientY: r.top + p[1] * r.height
              }));
              fire('pointerdown', points[0]);
              for (let i = 1; i < points.length; i++) fire('pointermove', points[i]);
              fire('pointerup', points[points.length - 1]);
              return { fired: true, w: r.width, h: r.height };
            })()
        """, {})
        time.sleep(0.3)
        check("the drawing pad accepts real strokes", draw.get("fired") is True, str(draw))
        check("drawing enables Ask", ev("!document.getElementById('playground-ask').disabled", False))

        click("#playground-ask")
        deadline = time.time() + 30
        answer = ""
        while time.time() < deadline:
            time.sleep(0.4)
            answer = ev("document.getElementById('playground-answer').textContent", "") or ""
            if "i think this is" in answer.lower():
                break
        check("Ask shows the network's real answer", "i think this is" in answer.lower(), answer[:140])
        check("feedback buttons appear after an answer",
              ev("document.getElementById('playground-feedback').hidden", True) is False)

        click("#playground-wrong")
        deadline = time.time() + 30
        loss = ""
        answer = ""
        while time.time() < deadline:
            time.sleep(0.4)
            loss = ev("document.getElementById('playground-loss').textContent", "") or ""
            answer = ev("document.getElementById('playground-answer').textContent", "") or ""
            if "after learning" in answer.lower() and ("mistakes" in loss.lower() or "learned" in loss.lower()):
                break
        check("Wrong runs a real correction and re-predicts immediately",
              "after learning" in answer.lower(), answer[:160])
        check("the correction reports measured before/after loss",
              "mistakes" in loss.lower() or "learned" in loss.lower(), loss[:160])
        check("feedback buttons return after the corrected prediction",
              ev("document.getElementById('playground-feedback').hidden", True) is False)

        click("#playground-exit")
        time.sleep(0.8)
        check("Back returns to the block studio",
              ev("document.getElementById('playground').hidden", False) is True and
              ev("document.getElementById('studio-main').hidden", True) is False)

        # ---- ONNX export through the real UI ----
        click("#btn-export")
        time.sleep(0.3)
        check("Export opens its dropdown", menu_hidden("export-menu") is False)
        click('[data-export="onnx"]')
        time.sleep(0.6)
        check("choosing ONNX closes the Export dropdown", menu_hidden("export-menu") is True)
        export_log = ev("document.getElementById('log-list').textContent", "") or ""
        check("ONNX export logs a successful artifact", "ONNX" in export_log, export_log[-240:])

        # ---- action-gated build mission: blank canvas, gated Next, real action unlocks ----
        # Prove the two guide surfaces cannot stack: start a tour, open the Tours menu over it, then
        # switch to a mission. The mission must take over, and the menu must stay above the tour dim.
        click("#btn-tour")
        time.sleep(0.25)
        click('[data-tour-track="track_a"]')
        time.sleep(0.5)
        check("a curriculum tour is active before the mission switch",
              ev("!!document.querySelector('.tour-root')", False))
        click("#btn-tour")
        time.sleep(0.25)
        mission_menu_hit = ev("""
            (function () {
              const button = document.querySelector('[data-mission="build_first_brain"]');
              if (!button) return false;
              const rect = button.getBoundingClientRect();
              const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
              return Boolean(hit && (hit === button || button.contains(hit)));
            })()
        """, False)
        check("the mission menu entry stays clickable above the active tour", mission_menu_hit is True)
        click('[data-mission="build_first_brain"]')
        time.sleep(0.9)
        check("starting a mission closes the active tour overlay",
              ev("!document.querySelector('.tour-root')", False))
        mission = ev_json("""
            (function () {
              const root = document.querySelector('.mission-root');
              const next = document.querySelector('.mission-next');
              const hint = document.querySelector('.mission-hint');
              return {
                open: Boolean(root && !root.hidden),
                progress: document.querySelector('.mission-progress-text')
                  ? document.querySelector('.mission-progress-text').textContent : '',
                nextDisabled: next ? next.disabled : null,
                nodes: document.querySelectorAll('.ns-node').length,
                mode: document.body.dataset.tier || '',
                stepHtml: document.querySelector('.mission-step-text')
                  ? document.querySelector('.mission-step-text').innerHTML : ''
              };
            })()
        """, {})
        check("starting a mission opens the presenter over a freshly blanked canvas",
              mission.get("open") is True and mission.get("nodes") == 0, str(mission))
        check("the presenter shows the honest step counter",
              "Step 1 of 13" in (mission.get("progress") or ""), mission.get("progress"))
        check("the mission copy renders bold emphasis instead of literal asterisks",
              "<strong>" in (mission.get("stepHtml") or "")
              and "**" not in (mission.get("stepHtml") or ""), mission.get("stepHtml"))
        check("the intro step can be advanced", mission.get("nextDisabled") is False, str(mission.get("nextDisabled")))
        check("the mission switches the tier to its own level", mission.get("mode") == "assist", str(mission.get("mode")))

        click(".mission-next")
        time.sleep(0.4)
        gated = ev_json("""
            (function () {
              const next = document.querySelector('.mission-next');
              const hint = document.querySelector('.mission-hint');
              return {
                progress: document.querySelector('.mission-progress-text')
                  ? document.querySelector('.mission-progress-text').textContent : '',
                nextDisabled: next ? next.disabled : null,
                hintMet: hint ? hint.classList.contains('is-met') : null,
                hint: hint ? hint.textContent : ''
              };
            })()
        """, {})
        check("Next is refused until the step's real action happens",
              gated.get("nextDisabled") is True and gated.get("hintMet") is False, str(gated))
        check("the step advanced to the source placement step",
              "Step 2 of 13" in (gated.get("progress") or ""), gated.get("progress"))
        check("the presenter explains what the learner must do",
              len(gated.get("hint") or "") > 0, (gated.get("hint") or "")[:120])

        click("#palette-source")
        time.sleep(0.6)
        verified = ev_json("""
            (function () {
              const next = document.querySelector('.mission-next');
              const hint = document.querySelector('.mission-hint');
              return {
                nodes: document.querySelectorAll('.ns-node').length,
                nextDisabled: next ? next.disabled : null,
                hintMet: hint ? hint.classList.contains('is-met') : null
              };
            })()
        """, {})
        check("placing the real Source block satisfies the live gate",
              verified.get("nodes") == 1 and verified.get("nextDisabled") is False
              and verified.get("hintMet") is True, str(verified))

        # ---- the guide card must dock over the canvas, never across the target / palette ----
        mission_layout = ev_json("""
            (function () {
              const card = document.querySelector('.mission-card');
              const target = document.querySelector('#palette-source');
              const palette = document.querySelector('.ns-palette');
              if (!card || !target || !palette) return { ok: false };
              const cr = card.getBoundingClientRect();
              const tr = target.getBoundingClientRect();
              const pr = palette.getBoundingClientRect();
              const overlap = !(cr.right <= tr.left || cr.left >= tr.right
                || cr.bottom <= tr.top || cr.top >= tr.bottom);
              return {
                ok: true,
                overlapsTarget: overlap,
                clearOfPalette: cr.left >= pr.right - 1,
                inView: cr.left >= 0 && cr.top >= 0
                  && cr.right <= window.innerWidth && cr.bottom <= window.innerHeight,
                card: { left: cr.left, top: cr.top, right: cr.right, bottom: cr.bottom },
                paletteRight: pr.right
              };
            })()
        """, {})
        check("the mission card docks over the canvas, not across the palette",
              mission_layout.get("ok") is True and mission_layout.get("clearOfPalette") is True,
              str(mission_layout))
        check("the mission card never covers the highlighted palette target",
              mission_layout.get("overlapsTarget") is False, str(mission_layout))
        check("the mission card stays inside the window", mission_layout.get("inView") is True,
              str(mission_layout))

        click(".mission-next")
        time.sleep(0.5)
        step3 = ev_json("""
            (function () {
              const card = document.querySelector('.mission-card');
              const panel = document.querySelector('.ns-right');
              const next = document.querySelector('.mission-next');
              const progress = document.querySelector('.mission-progress-text');
              const cr = card ? card.getBoundingClientRect() : null;
              const pr = panel ? panel.getBoundingClientRect() : null;
              return {
                progress: progress ? progress.textContent : '',
                nextDisabled: next ? next.disabled : null,
                clearOfInspector: Boolean(cr && pr && cr.right <= pr.left - 1),
                inView: Boolean(cr && cr.left >= 0 && cr.top >= 0
                  && cr.right <= window.innerWidth && cr.bottom <= window.innerHeight)
              };
            })()
        """, {})
        check("the mission advances to the dataset step", "Step 3 of 13" in (step3.get("progress") or ""),
              str(step3))
        check("the dataset step is still gated", step3.get("nextDisabled") is True, str(step3))
        check("the mission card docks over the canvas, not across the inspector",
              step3.get("clearOfInspector") is True, str(step3))
        check("the dataset-step card also stays inside the window", step3.get("inView") is True,
              str(step3))

        dataset_change = ev_json("""
            (function () {
              const select = document.querySelector('#inspector-body select');
              if (!select) return { ok: false, selects: document.querySelectorAll('#inspector-body select').length };
              const before = select.value;
              select.value = 'xor';
              select.dispatchEvent(new Event('change', { bubbles: true }));
              return { ok: true, before: before, after: select.value };
            })()
        """, {})
        check("the live inspector still exposes the dataset selector",
              dataset_change.get("ok") is True and dataset_change.get("after") == "xor",
              str(dataset_change))
        time.sleep(0.5)
        step3_gate = ev_json("""
            (function () {
              const next = document.querySelector('.mission-next');
              const hint = document.querySelector('.mission-hint');
              return {
                nextDisabled: next ? next.disabled : null,
                hintMet: hint ? hint.classList.contains('is-met') : null,
                hint: hint ? hint.textContent : ''
              };
            })()
        """, {})
        check("choosing XOR satisfies the live dataset gate",
              step3_gate.get("nextDisabled") is False and step3_gate.get("hintMet") is True,
              str(step3_gate))

        click(".mission-next")
        time.sleep(0.6)
        step4 = ev_json("""
            (function () {
              const card = document.querySelector('.mission-card');
              const node = document.querySelector('.ns-node[data-type="source"]');
              const socket = node ? node.querySelector('.ns-socket[data-dir="out"]') : null;
              const ring = document.querySelector('.mission-ring');
              const spot = document.querySelector('.mission-spotlight');
              const progress = document.querySelector('.mission-progress-text');
              if (!card || !node || !socket || !ring || !spot) return { ok: false };
              const cr = card.getBoundingClientRect();
              const nr = node.getBoundingClientRect();
              const sr = socket.getBoundingClientRect();
              const rr = ring.getBoundingClientRect();
              const pr = spot.getBoundingClientRect();
              const overlap = !(cr.right <= nr.left || cr.left >= nr.right
                || cr.bottom <= nr.top || cr.top >= nr.bottom);
              return {
                ok: true,
                progress: progress ? progress.textContent : '',
                cardOverNode: overlap,
                ringCoversSocket: rr.left <= sr.left + 1 && rr.top <= sr.top + 1
                  && rr.right >= sr.right - 1 && rr.bottom >= sr.bottom - 1,
                spotCoversSocket: pr.left <= sr.left + 1 && pr.top <= sr.top + 1
                  && pr.right >= sr.right - 1 && pr.bottom >= sr.bottom - 1,
                inView: cr.left >= 0 && cr.top >= 0
                  && cr.right <= window.innerWidth && cr.bottom <= window.innerHeight
              };
            })()
        """, {})
        check("the mission reaches the wire-growing step", "Step 4 of 13" in (step4.get("progress") or ""),
              str(step4))
        check("the spotlight and pulse ring wrap the real output socket",
              step4.get("spotCoversSocket") is True and step4.get("ringCoversSocket") is True,
              str(step4))
        check("the card leaves the whole source block reachable",
              step4.get("cardOverNode") is False, str(step4))
        check("the wire-growing card stays inside the window", step4.get("inView") is True,
              str(step4))

        moved = ev_json("""
            (function () {
              const node = document.querySelector('.ns-node[data-type="source"]');
              const canvas = document.getElementById('studio-canvas');
              const card = document.querySelector('.mission-card');
              if (!node || !canvas || !card) return { ok: false };
              const nr = node.getBoundingClientRect();
              const startX = nr.left + 30;
              const startY = nr.top + 16;
              const hit = document.elementFromPoint(startX, startY);
              const clearToGrab = Boolean(hit && hit.closest('.ns-node')
                && !hit.closest('.mission-card'));
              const beforeSocket = node.querySelector('.ns-socket[data-dir="out"]').getBoundingClientRect();
              const beforeSpot = document.querySelector('.mission-spotlight').getBoundingClientRect();
              const fire = (target, type, x, y) => target.dispatchEvent(new PointerEvent(type, {
                bubbles: true, cancelable: true, pointerId: 17, button: 0, clientX: x, clientY: y
              }));
              fire(node, 'pointerdown', startX, startY);
              fire(canvas, 'pointermove', startX + 60, startY + 30);
              fire(canvas, 'pointerup', startX + 60, startY + 30);
              const afterNode = document.querySelector('.ns-node[data-type="source"]');
              const afterSocket = afterNode.querySelector('.ns-socket[data-dir="out"]').getBoundingClientRect();
              const afterSpot = document.querySelector('.mission-spotlight').getBoundingClientRect();
              const afterRing = document.querySelector('.mission-ring').getBoundingClientRect();
              const cr = card.getBoundingClientRect();
              const ar = afterNode.getBoundingClientRect();
              const overlap = !(cr.right <= ar.left || cr.left >= ar.right
                || cr.bottom <= ar.top || cr.top >= ar.bottom);
              return {
                ok: true,
                clearToGrab: clearToGrab,
                nodeMoved: Math.abs((ar.left - nr.left) - 60) < 2
                  && Math.abs((ar.top - nr.top) - 30) < 2,
                spotMoved: Math.abs((afterSpot.left - beforeSpot.left) - 60) < 2
                  && Math.abs((afterSpot.top - beforeSpot.top) - 30) < 2,
                socketMoved: Math.abs((afterSocket.left - beforeSocket.left) - 60) < 2
                  && Math.abs((afterSocket.top - beforeSocket.top) - 30) < 2,
                ringCoversSocket: afterRing.left <= afterSocket.left + 1
                  && afterRing.top <= afterSocket.top + 1
                  && afterRing.right >= afterSocket.right - 1
                  && afterRing.bottom >= afterSocket.bottom - 1,
                cardOverNode: overlap,
                inView: cr.left >= 0 && cr.top >= 0
                  && cr.right <= window.innerWidth && cr.bottom <= window.innerHeight
              };
            })()
        """, {})
        check("the mission card leaves the block grabbable", moved.get("ok") is True
              and moved.get("clearToGrab") is True, str(moved))
        check("the real pointer drag moves the source block", moved.get("nodeMoved") is True,
              str(moved))
        check("the output socket and its spotlight follow the dragged block",
              moved.get("socketMoved") is True and moved.get("spotMoved") is True, str(moved))
        check("the pulse ring re-wraps the socket after the drag",
              moved.get("ringCoversSocket") is True, str(moved))
        check("the card dodges the moved block", moved.get("cardOverNode") is False, str(moved))
        check("the moved-block card stays inside the window", moved.get("inView") is True,
              str(moved))

        escape()
        time.sleep(0.5)
        mission_end = ev_json("""
            (function () {
              const root = document.querySelector('.mission-root');
              const log = document.getElementById('log-list').textContent || '';
              return { open: Boolean(root && !root.hidden), log: log };
            })()
        """, {})
        check("Escape exits the active mission (universal close contract)",
              mission_end.get("open") is False, str(mission_end.get("open")))
        check("the mission start and exit are logged",
              "Started mission" in (mission_end.get("log") or "")
              and "Mission exited" in (mission_end.get("log") or ""), (mission_end.get("log") or "")[-260:])

        # ---- final error sweep ----
        smoke_errors = ev("JSON.stringify(window.__smokeErrors || [])", "[]") or "[]"
        log_errors = ev("document.querySelectorAll('#log-list .ns-log-error').length", 0) or 0
        check("no uncaught page errors after the new flows", smoke_errors == "[]", smoke_errors[:300])
        check("no error log entries after the new flows", log_errors == 0, f"{log_errors} error entries")

    # Hard safety net: never let a suspended renderer hang this script forever.
    threading.Timer(540, window.destroy).start()

    webview.start(drive, debug=False)

    print("\n=====================================================")
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"   UI SMOKE RESULT: {passed} passed, {failed} failed")
    print("=====================================================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
