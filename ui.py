"""
Autonomous SRE Control Center — Industry-Grade AI Operations Interface.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from models import Action, CodeOrganismActionType
from environment import CodeOrganismEnv
from training.rollout import run_episode


CUSTOM_CSS = """
.gradio-container {
    background: #020617 !important;
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', 'Inter', sans-serif !important;
}
.sre-panel {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(12px);
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
}
.metric-label {
    color: #64748b;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 800;
    margin-bottom: 4px;
}
.terminal-output {
    background: #000 !important;
    border: 1px solid #0f172a !important;
    color: #38bdf8 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
}
.action-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    font-weight: 800 !important;
    border-radius: 8px !important;
}
.chaos-btn {
    background: linear-gradient(135deg, #f43f5e 0%, #9f1239 100%) !important;
    border: none !important;
    font-weight: 800 !important;
    color: white !important;
}
@keyframes pulse-emerald {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.sla-active { animation: pulse-emerald 2s infinite; }
"""


def _risk_color(risk: str) -> str:
    if risk == "Low":
        return "#10b981"
    if risk == "Medium":
        return "#f59e0b"
    return "#f43f5e"


def get_sla_html(vitality: float) -> str:
    color = "#10b981"
    status = "HEALTHY"
    if vitality < 30:
        color = "#f43f5e"
        status = "SLA BREACH"
    elif vitality < 70:
        color = "#f59e0b"
        status = "DEGRADED"

    return f"""
    <div class="sre-panel {'sla-active' if vitality > 80 else ''}" style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px;">
            <div>
                <div class="metric-label">System Health Index</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {color}; line-height: 1;">{vitality:.1f}%</div>
            </div>
            <div style="text-align: right;">
                <div class="metric-label">SLA Status</div>
                <div style="color: {color}; font-weight: 800; font-size: 0.9rem;">● {status}</div>
            </div>
        </div>
        <div style="width: 100%; background: #0f172a; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="width: {vitality}%; background: {color}; height: 100%; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
        </div>
    </div>
    """


def format_impact_html(downtime: float, confidence: float, risk: str) -> str:
    risk_color = _risk_color(risk)
    conf_color = "#10b981" if confidence > 0.8 else "#f59e0b"
    return f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div class="sre-panel" style="padding: 12px !important;">
            <div class="metric-label">Downtime Avoided</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #38bdf8;">{downtime:,.0f}s</div>
        </div>
        <div class="sre-panel" style="padding: 12px !important;">
            <div class="metric-label">AI Confidence</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: {conf_color};">{confidence * 100:.0f}%</div>
        </div>
        <div class="sre-panel" style="padding: 12px !important; grid-column: span 2;">
            <div class="metric-label">Operational Risk Assessment</div>
            <div style="font-size: 1rem; font-weight: 800; color: {risk_color};">{risk.upper()} PROBABILITY OF REGRESSION</div>
        </div>
    </div>
    """


def _format_diagnostics(test_results: list[Any]) -> str:
    diag_text = "### 🛡️ Deployment Diagnostics\n\n| Service | Status | Log |\n| :--- | :--- | :--- |\n"
    for test in test_results:
        icon = "✅" if test.status == "PASS" else "❌"
        diag_text += f"| {test.name} | {icon} {test.status} | {test.message or '--'} |\n"
    return diag_text


def _format_alerts(alerts: list[str]) -> str:
    return "".join(
        f"<div style='background: rgba(244, 63, 94, 0.15); border-left: 4px solid #f43f5e; padding: 10px; margin-bottom: 8px; font-size: 0.8rem; font-weight: bold;'>{alert}</div>"
        for alert in alerts
    )


def _format_episode_postmortem(trace: dict[str, Any]) -> str:
    actions = trace.get("actions", [])[:8]
    timeline = "\n".join(
        f"- step {a['step']}: `{a['action_type']}` reward={a['reward']}"
        for a in actions
    ) or "- no actions logged"
    return (
        "### 📄 Episode Postmortem\n"
        f"- Policy: `{trace.get('policy')}`\n"
        f"- Task: `{trace.get('task_id')}`\n"
        f"- Seed: `{trace.get('seed')}`\n"
        f"- Termination: `{trace.get('termination')}`\n"
        f"- Survived: `{trace.get('survived')}`\n"
        f"- Total Reward: `{trace.get('total_reward')}`\n"
        f"- Final Vitality: `{trace.get('final_vitality')}`\n"
        "\n### ⏱ Timeline\n"
        f"{timeline}"
    )


def reset_center(env: CodeOrganismEnv, task_id: str):
    obs = env.reset(task_id)
    return (
        get_sla_html(obs.vitality_score),
        format_impact_html(0, 0, "N/A"),
        "### 🔎 Diagnostics\nInitializing telemetry...\nREADY.",
        f"**CLUSTER:** {task_id.upper()} | **ID:** {env._episode_id}",
        "Waiting for incident...",
        obs.dependency_graph,
        obs.recent_signals,
        [],
        "### 📄 Episode Postmortem\nNo episode replay yet.",
    )


def trigger_chaos(env: CodeOrganismEnv):
    msg = env.inject_chaos()
    state = env.state()
    return (
        get_sla_html(state.vitality),
        f"### 🚨 CHAOS ENGINE ACTIVATED\n{msg}\nInjecting failure vectors...",
        [],
    )


def run_demo_episode(task_id: str, policy: str):
    seed = 104857 if task_id == "phase_3" else 11000
    trace = run_episode(policy=policy, task_id=task_id, seed=seed).to_dict()
    return (
        _format_episode_postmortem(trace),
        f"### ✅ Demo run complete\nPolicy `{policy}` on `{task_id}` finished.",
    )


def process_protocol(
    env: CodeOrganismEnv,
    action_type: str,
    path: str,
    diff: str,
    sub_task: str,
    checkpoint_id: str,
    query: str,
    signal_type: str,
    justification: str,
):
    try:
        action = Action(
            action_type=CodeOrganismActionType(action_type),
            path=path or None,
            diff=diff or None,
            task=sub_task or None,
            checkpoint_id=checkpoint_id or None,
            query=query or None,
            signal_type=signal_type or None,
            signal_data={"target": path} if signal_type == "INTENT_PATCH" else None,
            justification=justification or "",
        )
        result = env.step(action)
    except Exception as exc:
        return None, None, f"### ⚠️ PROTOCOL ERROR\n{exc}", "IDLE", "FAILURE", {}, [], "", "### 📄 Episode Postmortem\nProtocol failed."

    obs = result.observation or env._make_observation()
    state = env.state()
    sre = result.info.get("sre_metrics", {"confidence": 0, "risk_assessment": "High", "downtime_saved_total": 0})
    status_line = f"**STEP:** {obs.timestep} | **CUMULATIVE_EFFICIENCY:** {state.cumulative_reward:.4f}"
    if result.done:
        status_line = f"### 🏁 SESSION COMPLETE | FINAL_REWARD: {state.cumulative_reward:.4f}"
    return (
        get_sla_html(obs.vitality_score),
        format_impact_html(sre["downtime_saved_total"], sre["confidence"], sre["risk_assessment"]),
        _format_diagnostics(obs.test_results),
        status_line,
        obs.stack_trace or "No active stack traces.",
        obs.dependency_graph,
        obs.recent_signals,
        _format_alerts(obs.alerts or []),
        result.info.get("postmortem") or "### 📄 Episode Postmortem\nIn progress...",
    )


def create_gradio_app() -> gr.Blocks:
    env = CodeOrganismEnv()
    with gr.Blocks(title="Autonomous SRE Control Center", css=CUSTOM_CSS) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🛰️ Autonomous SRE Control Center")
                gr.Markdown("*Self-Healing Infrastructure Dashboard | Powered by Meta OpenEnv*")
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(["phase_1", "phase_2", "phase_3"], value="phase_1", label="Incident Difficulty Profile")
                reset_btn = gr.Button("PROVISION CLUSTER", variant="secondary", elem_classes=["action-btn"])

        with gr.Row():
            with gr.Column(scale=2):
                sla_display = gr.HTML(get_sla_html(100))
                impact_display = gr.HTML(format_impact_html(0, 0, "Low"))
            with gr.Column(scale=1, elem_classes=["sre-panel"]):
                gr.Markdown("<div class='metric-label'>System Log Feed</div>")
                status_bar = gr.Markdown("Systems Standby.")
                chaos_btn = gr.Button("🔥 TRIGGER CHAOS INCIDENT", elem_classes=["chaos-btn"])
                run_noop_btn = gr.Button("▶️ RUN BASELINE EPISODE", elem_classes=["action-btn"])
                run_heuristic_btn = gr.Button("▶️ RUN HEURISTIC EPISODE", elem_classes=["action-btn"])

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["sre-panel"]):
                    gr.Markdown("<div class='metric-label'>🚨 Incident Alerts</div>")
                    alerts_display = gr.HTML("")
                    gr.Markdown("<div class='metric-label'>📡 Node Signals</div>")
                    signals_display = gr.JSON(label=None, show_label=False)
                with gr.Column(elem_classes=["sre-panel"]):
                    gr.Markdown("<div class='metric-label'>🏗️ Deployment Topology</div>")
                    world_model_display = gr.JSON(label=None, show_label=False)

            with gr.Column(scale=2):
                with gr.Tabs(elem_classes=["sre-panel"]):
                    with gr.Tab("Remediation Protocol"):
                        gr.Markdown("### 🕹 COMMAND CONSOLE")
                        with gr.Row():
                            action_type = gr.Dropdown([e.value for e in CodeOrganismActionType], value="patch_file", label="OP_CODE")
                            path_box = gr.Textbox(label="TARGET_NODE", placeholder="src/core.py")
                            signal_box = gr.Textbox(label="SIGNAL_METADATA", placeholder="INTENT_PATCH")
                        with gr.Row():
                            diff_box = gr.Textbox(label="REMEDIATION_PAYLOAD", lines=3, placeholder="OLD|NEW")
                            justification_box = gr.Textbox(label="PROTOCOL_JUSTIFICATION", lines=3)
                        with gr.Row():
                            sub_task_box = gr.Textbox(label="DELEGATED_SUBTASK")
                            query_box = gr.Textbox(label="EXPERT_ORACLE_QUERY")
                            checkpoint_box = gr.Textbox(label="RESTORE_POINT_ID")
                        submit_btn = gr.Button("EXECUTE REMEDIATION", variant="primary", elem_classes=["action-btn"])

                    with gr.Tab("Active Diagnostics"):
                        test_display = gr.Markdown("TELEMETRY_IDLE")
                    with gr.Tab("Error Trace"):
                        stack_display = gr.Markdown("", elem_classes=["terminal-output"])
                    with gr.Tab("Episode Postmortem"):
                        postmortem_display = gr.Markdown("### 📄 Episode Postmortem\nNo episode replay yet.")

        reset_btn.click(
            lambda task_id: reset_center(env, task_id),
            inputs=[task_dd],
            outputs=[sla_display, impact_display, test_display, status_bar, stack_display, world_model_display, signals_display, alerts_display, postmortem_display],
        )
        chaos_btn.click(lambda: trigger_chaos(env), outputs=[sla_display, stack_display, alerts_display])
        run_noop_btn.click(
            lambda task_id: run_demo_episode(task_id, "noop"),
            inputs=[task_dd],
            outputs=[postmortem_display, status_bar],
        )
        run_heuristic_btn.click(
            lambda task_id: run_demo_episode(task_id, "heuristic"),
            inputs=[task_dd],
            outputs=[postmortem_display, status_bar],
        )
        submit_btn.click(
            lambda action_type, path, diff, sub_task, checkpoint_id, query, signal_type, justification: process_protocol(
                env, action_type, path, diff, sub_task, checkpoint_id, query, signal_type, justification
            ),
            inputs=[action_type, path_box, diff_box, sub_task_box, checkpoint_box, query_box, signal_box, justification_box],
            outputs=[sla_display, impact_display, test_display, status_bar, stack_display, world_model_display, signals_display, alerts_display, postmortem_display],
        )
    return demo
