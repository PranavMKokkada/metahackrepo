"""
Autonomous SRE Control Center — Industry-Grade AI Operations Interface.
"""

from __future__ import annotations

import time
from typing import Any

import gradio as gr

from models import Action, CodeOrganismActionType
from environment import CodeOrganismEnv
from training.rollout import run_episode


CUSTOM_CSS = """
.gradio-container {
    background: #0a0a0a !important;
    color: #f5f5f5 !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
    letter-spacing: 0.02em;
}
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose span {
    color: #d4d4d4 !important;
}
.gradio-container .gr-button,
.gradio-container button {
    border-radius: 10px !important;
    border: 1px solid #2f2f2f !important;
    background: #171717 !important;
    color: #f5f5f5 !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.gradio-container .gr-button:hover,
.gradio-container button:hover {
    border-color: #525252 !important;
    background: #1f1f1f !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container .gr-dropdown,
.gradio-container .gr-textbox,
.gradio-container .gr-json {
    background: #111111 !important;
    color: #f5f5f5 !important;
    border: 1px solid #2c2c2c !important;
}
.gradio-container .gr-tab-nav button {
    background: #121212 !important;
    color: #d4d4d4 !important;
    border: 1px solid #262626 !important;
}
.gradio-container .gr-tab-nav button.selected {
    color: #ffffff !important;
    border-color: #525252 !important;
    background: #1a1a1a !important;
}
.sre-panel {
    background: #121212 !important;
    border: 1px solid #262626 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.metric-label {
    color: #a3a3a3;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 800;
    margin-bottom: 4px;
}
.terminal-output {
    background: #090909 !important;
    border: 1px solid #262626 !important;
    color: #f5f5f5 !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 0.85rem !important;
}
.action-btn {
    background: #171717 !important;
    border: 1px solid #2f2f2f !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    color: #f5f5f5 !important;
}
.chaos-btn {
    background: #1b1b1b !important;
    border: 1px solid #3b3b3b !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
.sla-active { border-color: #4a4a4a !important; }
"""


def _risk_color(risk: str) -> str:
    if risk == "Low":
        return "#d4d4d4"
    if risk == "Medium":
        return "#a3a3a3"
    return "#ffffff"


def get_sla_html(vitality: float) -> str:
    color = "#f5f5f5"
    status = "HEALTHY"
    if vitality < 30:
        color = "#ffffff"
        status = "SLA BREACH"
    elif vitality < 70:
        color = "#d4d4d4"
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
        <div style="width: 100%; background: #1a1a1a; border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="width: {vitality}%; background: {color}; height: 100%; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
        </div>
    </div>
    """


def format_impact_html(downtime: float, confidence: float, risk: str) -> str:
    risk_color = _risk_color(risk)
    conf_color = "#f5f5f5" if confidence > 0.8 else "#d4d4d4"
    return f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div class="sre-panel" style="padding: 12px !important;">
            <div class="metric-label">Downtime Avoided</div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #f5f5f5;">{downtime:,.0f}s</div>
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
    diag_text = "### Deployment Diagnostics\n\n| Service | Status | Log |\n| :--- | :--- | :--- |\n"
    for test in test_results:
        icon = "✅" if test.status == "PASS" else "❌"
        diag_text += f"| {test.name} | {icon} {test.status} | {test.message or '--'} |\n"
    return diag_text


def _format_alerts(alerts: list[str]) -> str:
    return "".join(
        f"<div style='background: #161616; border-left: 3px solid #5a5a5a; padding: 10px; margin-bottom: 8px; font-size: 0.8rem; font-weight: 600;'>{alert}</div>"
        for alert in alerts
    )


def _format_episode_postmortem(trace: dict[str, Any]) -> str:
    actions = trace.get("actions", [])[:8]
    timeline = "\n".join(
        f"- step {a['step']}: `{a['action_type']}` reward={a['reward']}"
        for a in actions
    ) or "- no actions logged"
    return (
        "### Episode Postmortem\n"
        f"- Policy: `{trace.get('policy')}`\n"
        f"- Task: `{trace.get('task_id')}`\n"
        f"- Seed: `{trace.get('seed')}`\n"
        f"- Termination: `{trace.get('termination')}`\n"
        f"- Survived: `{trace.get('survived')}`\n"
        f"- Total Reward: `{trace.get('total_reward')}`\n"
        f"- Final Vitality: `{trace.get('final_vitality')}`\n"
        "\n### Timeline\n"
        f"{timeline}"
    )


def reset_center(env: CodeOrganismEnv, task_id: str):
    obs = env.reset(task_id)
    return (
        get_sla_html(obs.vitality_score),
        format_impact_html(0, 0, "N/A"),
        "### Diagnostics\nInitializing telemetry...\nREADY.",
        f"**CLUSTER:** {task_id.upper()} | **ID:** {env._episode_id}",
        "Waiting for incident...",
        obs.dependency_graph,
        obs.recent_signals,
        [],
        "### Episode Postmortem\nNo episode replay yet.",
    )


def trigger_chaos(env: CodeOrganismEnv):
    msg = env.inject_chaos()
    state = env.state()
    return (
        get_sla_html(state.vitality),
        f"### CHAOS ENGINE ACTIVATED\n{msg}\nInjecting failure vectors...",
        [],
    )


def run_demo_episode(task_id: str, policy: str):
    seed = 104857 if task_id == "phase_3" else 11000
    trace = run_episode(policy=policy, task_id=task_id, seed=seed).to_dict()
    return (
        _format_episode_postmortem(trace),
        f"### Demo run complete\nPolicy `{policy}` on `{task_id}` finished.",
    )


def run_guided_demo(env: CodeOrganismEnv, task_id: str):
    """Auto-narrated walkthrough for recording a polished demo."""
    obs = env.reset(task_id)
    intro = (
        "### Guided Demo Mode\n"
        "**Stage 1 - Situation Awareness**\n"
        "A fresh incident scenario is provisioned. The agent inspects health, diagnostics, and the service graph before acting."
    )
    yield (
        get_sla_html(obs.vitality_score),
        format_impact_html(0, 0, "N/A"),
        _format_diagnostics(obs.test_results),
        f"**CLUSTER:** {task_id.upper()} | **ID:** {env._episode_id} | **MODE:** GUIDED_DEMO",
        obs.stack_trace or "No active stack traces.",
        obs.dependency_graph,
        obs.recent_signals,
        _format_alerts(obs.alerts or []),
        "### Episode Postmortem\nGuided demo started.",
        intro,
    )
    time.sleep(0.8)

    chaos_msg = env.inject_chaos()
    obs = env._make_observation()
    yield (
        get_sla_html(obs.vitality_score),
        format_impact_html(0, 0, "High"),
        _format_diagnostics(obs.test_results),
        "**MODE:** GUIDED_DEMO | Chaos injected intentionally",
        chaos_msg,
        obs.dependency_graph,
        obs.recent_signals,
        _format_alerts(obs.alerts or []),
        "### Episode Postmortem\nChaos event introduced for controlled demonstration.",
        (
            "### Guided Demo Mode\n"
            "**Stage 2 - Controlled Failure Injection**\n"
            "The chaos engine injects live faults to show this is not a static UI replay."
        ),
    )
    time.sleep(0.8)

    scripted_actions = [
        (
            "Signal remediation intent",
            Action(
                action_type=CodeOrganismActionType.EMIT_SIGNAL,
                signal_type="INTENT_PATCH",
                signal_data={"target": "src/core.py"},
                justification="Broadcasting repair intent for coordinated execution.",
            ),
            "The agent announces intent first, earning coordination behavior and reducing blind actions.",
        ),
        (
            "Run diagnostics",
            Action(
                action_type=CodeOrganismActionType.RUN_TESTS,
                justification="Gather fresh test status before changing code.",
            ),
            "Diagnostics establish ground truth and prevent random patching.",
        ),
        (
            "Patch a known failure pattern",
            Action(
                action_type=CodeOrganismActionType.PATCH_FILE,
                path="src/core.py",
                diff="retunr|return",
                justification="Apply deterministic hotfix for syntax corruption.",
            ),
            "A surgical patch is applied to recover execution safety.",
        ),
        (
            "Delegate parallel stabilization",
            Action(
                action_type=CodeOrganismActionType.SPAWN_SUBAGENT,
                task="stabilize dependent services and clear residual regressions",
                justification="Parallelize remediation under time pressure.",
            ),
            "Subagent delegation demonstrates multi-agent coordination in long-horizon recovery.",
        ),
        (
            "Request expert validation",
            Action(
                action_type=CodeOrganismActionType.REQUEST_EXPERT,
                query="Validate whether the latest patch is safe and production-worthy.",
                justification="Safety gate before declaring recovery.",
            ),
            "Expert feedback closes the loop by checking patch quality, not just immediate test pass rate.",
        ),
        (
            "Re-run tests for verification",
            Action(
                action_type=CodeOrganismActionType.RUN_TESTS,
                justification="Confirm that remediation improved runtime behavior.",
            ),
            "Final verification demonstrates measurable behavior change after intervention.",
        ),
    ]

    for idx, (title, action, explanation) in enumerate(scripted_actions, start=1):
        result = env.step(action)
        obs = result.observation or env._make_observation()
        state = env.state()
        sre = result.info.get(
            "sre_metrics",
            {"confidence": 0.0, "risk_assessment": "High", "downtime_saved_total": 0.0},
        )
        status_line = (
            f"**MODE:** GUIDED_DEMO | **STEP:** {obs.timestep} | "
            f"**ACTION:** {action.action_type.value} | **REWARD:** {result.reward:.4f}"
        )
        if result.done:
            status_line = f"### SESSION COMPLETE | FINAL_REWARD: {state.cumulative_reward:.4f}"

        narrative = (
            "### Guided Demo Mode\n"
            f"**Stage 3.{idx} - {title}**\n"
            f"{explanation}\n\n"
            f"- Action: `{action.action_type.value}`\n"
            f"- Reward (step): `{result.reward:.4f}`\n"
            f"- Cumulative reward: `{state.cumulative_reward:.4f}`\n"
            f"- Vitality: `{obs.vitality_score:.1f}%`"
        )
        yield (
            get_sla_html(obs.vitality_score),
            format_impact_html(
                sre.get("downtime_saved_total", 0),
                sre.get("confidence", 0),
                sre.get("risk_assessment", "High"),
            ),
            _format_diagnostics(obs.test_results),
            status_line,
            obs.stack_trace or "No active stack traces.",
            obs.dependency_graph,
            obs.recent_signals,
            _format_alerts(obs.alerts or []),
            result.info.get("postmortem") or "### Episode Postmortem\nGuided demo in progress.",
            narrative,
        )
        time.sleep(0.8)
        if result.done:
            break


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
        return None, None, f"### PROTOCOL ERROR\n{exc}", "IDLE", "FAILURE", {}, [], "", "### Episode Postmortem\nProtocol failed."

    obs = result.observation or env._make_observation()
    state = env.state()
    sre = result.info.get("sre_metrics", {"confidence": 0, "risk_assessment": "High", "downtime_saved_total": 0})
    status_line = f"**STEP:** {obs.timestep} | **CUMULATIVE_EFFICIENCY:** {state.cumulative_reward:.4f}"
    if result.done:
        status_line = f"### SESSION COMPLETE | FINAL_REWARD: {state.cumulative_reward:.4f}"
    return (
        get_sla_html(obs.vitality_score),
        format_impact_html(sre["downtime_saved_total"], sre["confidence"], sre["risk_assessment"]),
        _format_diagnostics(obs.test_results),
        status_line,
        obs.stack_trace or "No active stack traces.",
        obs.dependency_graph,
        obs.recent_signals,
        _format_alerts(obs.alerts or []),
        result.info.get("postmortem") or "### Episode Postmortem\nIn progress...",
    )


def create_gradio_app() -> gr.Blocks:
    env = CodeOrganismEnv()
    with gr.Blocks(title="Autonomous SRE Control Center", css=CUSTOM_CSS) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# Autonomous SRE Control Center")
                gr.Markdown("**Self-Healing Infrastructure Dashboard**")
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(["phase_1", "phase_2", "phase_3"], value="phase_1", label="Incident Profile")
                reset_btn = gr.Button("Initialize Session", variant="secondary", elem_classes=["action-btn"])

        with gr.Row():
            with gr.Column(scale=2):
                sla_display = gr.HTML(get_sla_html(100))
                impact_display = gr.HTML(format_impact_html(0, 0, "Low"))
            with gr.Column(scale=1, elem_classes=["sre-panel"]):
                gr.Markdown("<div class='metric-label'>System Log Feed</div>")
                status_bar = gr.Markdown("Systems Standby.")
                chaos_btn = gr.Button("Trigger Chaos Incident", elem_classes=["chaos-btn"])
                run_noop_btn = gr.Button("Run Baseline Episode", elem_classes=["action-btn"])
                run_heuristic_btn = gr.Button("Run Heuristic Episode", elem_classes=["action-btn"])
                run_guided_btn = gr.Button("Run Guided Demo Mode", elem_classes=["action-btn"])

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["sre-panel"]):
                    gr.Markdown("<div class='metric-label'>Incident Alerts</div>")
                    alerts_display = gr.HTML("")
                    gr.Markdown("<div class='metric-label'>Node Signals</div>")
                    signals_display = gr.JSON(label=None, show_label=False)
                with gr.Column(elem_classes=["sre-panel"]):
                    gr.Markdown("<div class='metric-label'>Deployment Topology</div>")
                    world_model_display = gr.JSON(label=None, show_label=False)

            with gr.Column(scale=2):
                guided_demo_brief = gr.Markdown(
                    "### Guided Demo Mode\nUse this to auto-run a narrated, judge-friendly walkthrough."
                )
                with gr.Tabs(elem_classes=["sre-panel"]):
                    with gr.Tab("Remediation Protocol"):
                        gr.Markdown("### Command Console")
                        with gr.Row():
                            action_type = gr.Dropdown([e.value for e in CodeOrganismActionType], value="patch_file", label="Operation")
                            path_box = gr.Textbox(label="Target Path", placeholder="src/core.py")
                            signal_box = gr.Textbox(label="Signal Metadata", placeholder="INTENT_PATCH")
                        with gr.Row():
                            diff_box = gr.Textbox(label="Remediation Payload", lines=3, placeholder="OLD|NEW")
                            justification_box = gr.Textbox(label="Justification", lines=3)
                        with gr.Row():
                            sub_task_box = gr.Textbox(label="Delegated Subtask")
                            query_box = gr.Textbox(label="Expert Oracle Query")
                            checkpoint_box = gr.Textbox(label="Restore Point ID")
                        submit_btn = gr.Button("Execute Remediation", variant="primary", elem_classes=["action-btn"])

                    with gr.Tab("Active Diagnostics"):
                        test_display = gr.Markdown("TELEMETRY_IDLE")
                    with gr.Tab("Error Trace"):
                        stack_display = gr.Markdown("", elem_classes=["terminal-output"])
                    with gr.Tab("Episode Postmortem"):
                        postmortem_display = gr.Markdown("### Episode Postmortem\nNo episode replay yet.")

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
        run_guided_btn.click(
            lambda task_id: run_guided_demo(env, task_id),
            inputs=[task_dd],
            outputs=[
                sla_display,
                impact_display,
                test_display,
                status_bar,
                stack_display,
                world_model_display,
                signals_display,
                alerts_display,
                postmortem_display,
                guided_demo_brief,
            ],
        )
        submit_btn.click(
            lambda action_type, path, diff, sub_task, checkpoint_id, query, signal_type, justification: process_protocol(
                env, action_type, path, diff, sub_task, checkpoint_id, query, signal_type, justification
            ),
            inputs=[action_type, path_box, diff_box, sub_task_box, checkpoint_box, query_box, signal_box, justification_box],
            outputs=[sla_display, impact_display, test_display, status_bar, stack_display, world_model_display, signals_display, alerts_display, postmortem_display],
        )
    return demo
