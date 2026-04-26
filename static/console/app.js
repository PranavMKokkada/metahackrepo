(() => {
  const $ = (id) => document.getElementById(id);

  function headers() {
    const k = localStorage.getItem("sre_api_key") || $("apiKey").value.trim();
    const h = { "Content-Type": "application/json" };
    if (k) h["x-api-key"] = k;
    return h;
  }

  async function api(path, opts = {}) {
    const r = await fetch(path, { ...opts, headers: { ...headers(), ...opts.headers } });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    const ct = r.headers.get("content-type") || "";
    if (ct.includes("application/json")) return r.json();
    return r.text();
  }

  function showTab(name) {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
    document.querySelectorAll("#panels > section").forEach((p) => {
      p.classList.toggle("hidden", p.id !== `tab-${name}`);
    });
  }

  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => showTab(t.dataset.tab));
  });

  $("saveKey").addEventListener("click", () => {
    localStorage.setItem("sre_api_key", $("apiKey").value.trim());
    refreshAll();
  });

  let lastStep = null;

  async function refreshPlatform() {
    const data = await api("/platform/session/state");
    $("episodeBox").textContent = JSON.stringify(data.episode, null, 2);
    $("cicdBox").textContent = JSON.stringify(data.cicd, null, 2);
    $("bizBox").textContent = JSON.stringify(data.business, null, 2);
    $("predictBox").textContent = JSON.stringify(data.predictive, null, 2);
    $("evoBox").textContent = JSON.stringify(data.evolution, null, 2);
    $("memoryBox").textContent = JSON.stringify(data.memory_recent, null, 2);
    $("prodToggle").checked = !!data.production_mode;
    $("rollbackMin").value = String(data.guardrails.rollback_confidence_min ?? 0.55);
    $("restrictedExtra").value = (data.guardrails.restricted_paths_extra || []).join(",");
    $("safeZones").value = (data.guardrails.safe_zones || []).join(",");
    $("catBlock").checked = !!data.guardrails.catastrophic_block;
    renderSuggestions(data.pending_suggestion_batch);
    if (lastStep && lastStep.explanation) {
      $("explainBox").textContent = JSON.stringify(lastStep.explanation, null, 2);
    }
    if (lastStep && lastStep.specialized_agent) {
      $("agentsBox").textContent = JSON.stringify(lastStep.specialized_agent, null, 2);
    }
  }

  function renderSuggestions(batch) {
    const box = $("suggestionsBox");
    const row = $("approveRow");
    box.innerHTML = "";
    row.innerHTML = "";
    if (!batch || !batch.suggestions) return;
    batch.suggestions.forEach((s) => {
      const d = document.createElement("div");
      d.className = "sug";
      d.innerHTML = `<div><strong>#${s.rank}</strong> score ${s.score}<div class="meta">${s.rationale}</div><pre class="mono small">${s.path}\n${s.diff}</pre></div>`;
      const b = document.createElement("button");
      b.textContent = "Approve";
      b.addEventListener("click", async () => {
        await api("/platform/session/production/approve", {
          method: "POST",
          body: JSON.stringify({ suggestion_id: s.suggestion_id }),
        });
        await refreshPlatform();
      });
      d.appendChild(b);
      box.appendChild(d);
    });
    const rej = document.createElement("button");
    rej.textContent = "Reject all";
    rej.addEventListener("click", async () => {
      await api("/platform/session/production/reject", { method: "POST", body: JSON.stringify({}) });
      await refreshPlatform();
    });
    row.appendChild(rej);
  }

  async function refreshAll() {
    try {
      await refreshPlatform();
    } catch (e) {
      $("episodeBox").textContent = String(e);
    }
  }

  $("btnRefresh").addEventListener("click", refreshAll);

  $("btnReset").addEventListener("click", async () => {
    const obs = await api("/reset", { method: "POST", body: JSON.stringify({ task_id: "phase_1" }) });
    $("lastStepBox").textContent = JSON.stringify({ reset: true, observation: obs }, null, 2);
    await refreshAll();
  });

  $("btnRunTests").addEventListener("click", async () => {
    lastStep = await api("/step", {
      method: "POST",
      body: JSON.stringify({ action_type: "run_tests", justification: "Console smoke" }),
    });
    $("lastStepBox").textContent = JSON.stringify(lastStep, null, 2);
    await refreshAll();
  });

  $("prodToggle").addEventListener("change", async () => {
    await api("/platform/session/production-mode", {
      method: "POST",
      body: JSON.stringify({ enabled: $("prodToggle").checked }),
    });
    await refreshAll();
  });

  $("btnSaveGuard").addEventListener("click", async () => {
    const body = {
      rollback_confidence_min: parseFloat($("rollbackMin").value),
      restricted_paths_extra: $("restrictedExtra").value.split(",").map((s) => s.trim()).filter(Boolean),
      safe_zones: $("safeZones").value.split(",").map((s) => s.trim()).filter(Boolean),
      catastrophic_block: $("catBlock").checked,
    };
    await api("/platform/session/guardrails", { method: "POST", body: JSON.stringify(body) });
    await refreshAll();
  });

  $("btnMemoryLookup").addEventListener("click", async () => {
    const m = await api("/platform/session/memory/lookup");
    $("memoryBox").textContent = JSON.stringify(m, null, 2);
  });

  $("btnIngest").addEventListener("click", async () => {
    const lines = $("logLines").value.split("\n").filter(Boolean);
    await api("/platform/session/logs/ingest", {
      method: "POST",
      body: JSON.stringify({ lines, source: "operator_console" }),
    });
    await refreshAll();
  });

  const key = localStorage.getItem("sre_api_key");
  if (key) $("apiKey").value = key;
  refreshAll();
})();
