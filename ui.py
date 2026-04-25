"""
Autonomous SRE Control Center — Industry-Grade AI Operations Interface.

Pivots the CodeOrganismVM into a professional Site Reliability Engineering 
dashboard featuring SLA telemetry, Chaos Engineering triggers, and 
Explainable AI (XAI) guardrails.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Any

import gradio as gr

from models import Action, CodeOrganismActionType
from environment import CodeOrganismEnv, VITALITY_COSTS


def create_gradio_app() -> gr.Blocks:
    env = CodeOrganismEnv()

    # ── Premium SRE Design System ──────────────────────────────────────────────

    custom_css = """
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

    /* Status Animations */
    @keyframes pulse-emerald {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .sla-active { animation: pulse-emerald 2s infinite; }
    """

    def get_sla_html(v: float) -> str:
        color = "#10b981" # Emerald
        status = "HEALTHY"
        if v < 30:
            color = "#f43f5e" # Rose
            status = "SLA BREACH"
        elif v < 70:
            color = "#f59e0b" # Amber
            status = "DEGRADED"
        
        return f'''
        <div class="sre-panel {'sla-active' if v > 80 else ''}" style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 10px;">
                <div>
                    <div class="metric-label">System Health Index</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {color}; line-height: 1;">{v:.1f}%</div>
                </div>
                <div style="text-align: right;">
                    <div class="metric-label">SLA Status</div>
                    <div style="color: {color}; font-weight: 800; font-size: 0.9rem;">● {status}</div>
                </div>
            </div>
            <div style="width: 100%; background: #0f172a; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="width: {v}%; background: {color}; height: 100%; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
        </div>
        '''

    def format_impact_html(downtime: float, confidence: float, risk: str) -> str:
        risk_color = "#10b981" if risk == "Low" else "#f59e0b" if risk == "Medium" else "#f43f5e"
        conf_color = "#10b981" if confidence > 0.8 else "#f59e0b"
        
        return f'''
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="sre-panel" style="padding: 12px !important;">
                <div class="metric-label">Downtime Avoided</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #38bdf8;">{downtime:,.0f}s</div>
            </div>
            <div class="sre-panel" style="padding: 12px !important;">
                <div class="metric-label">AI Confidence</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: {conf_color};">{confidence*100:.0f}%</div>
            </div>
            <div class="sre-panel" style="padding: 12px !important; grid-column: span 2;">
                <div class="metric-label">Operational Risk Assessment</div>
                <div style="font-size: 1rem; font-weight: 800; color: {risk_color};">{risk.upper()} PROBABILITY OF REGRESSION</div>
            </div>
        </div>
        '''

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def reset_center(task_id):
        obs = env.reset(task_id)
        return (
            get_sla_html(obs.vitality_score),
            format_impact_html(0, 0, "N/A"),
            "### 🔎 Diagnostics\nInitializing telemetry...\nREADY.",
            f"**CLUSTER:** {task_id.upper()} | **ID:** {env._episode_id}",
            "Waiting for incident...",
            obs.dependency_graph,
            obs.recent_signals,
            []
        )

    def trigger_chaos():
        msg = env.inject_chaos()
        st = env.state()
        return (
            get_sla_html(st.vitality),
            f"### 🚨 CHAOS ENGINE ACTIVATED\n{msg}\nInjecting failure vectors...",
            [] # Refresh alerts
        )

    def process_protocol(
        action_type, path, diff, sub_task,
        checkpoint_id, query, signal_type, justification
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
                justification=justification or ""
            )
            result = env.step(action)
        except Exception as e:
            return None, None, f"### ⚠️ PROTOCOL ERROR\n{e}", "IDLE", "FAILURE", {}, [], ""

        obs = result.observation
        st = env.state()
        sre = result.info.get("sre_metrics", {"confidence": 0, "risk_assessment": "High", "downtime_saved_total": 0})
        
        diag_text = "### 🛡️ Deployment Diagnostics\n\n| Service | Status | Log |\n| :--- | :--- | :--- |\n"
        for t in obs.test_results:
            icon = "✅" if t.status == "PASS" else "❌"
            diag_text += f"| {t.name} | {icon} {t.status} | {t.message or '--'} |\n"

        alert_html = ""
        if obs.alerts:
            for a in obs.alerts:
                alert_html += f"<div style='background: rgba(244, 63, 94, 0.15); border-left: 4px solid #f43f5e; padding: 10px; margin-bottom: 8px; font-size: 0.8rem; font-weight: bold;'>{a}</div>"

        status_line = f"**STEP:** {obs.timestep} | **CUMULATIVE_EFFICIENCY:** {st.cumulative_reward:.4f}"
        if result.done:
            status_line = f"### 🏁 SESSION COMPLETE | FINAL_REWARD: {st.cumulative_reward:.4f}"

        return (
            get_sla_html(obs.vitality_score),
            format_impact_html(sre['downtime_saved_total'], sre['confidence'], sre['risk_assessment']),
            diag_text,
            status_line,
            obs.stack_trace or "No active stack traces.",
            obs.dependency_graph,
            obs.recent_signals,
            alert_html
        )

    # ── Main Layout ──────────────────────────────────────────────────────────

    with gr.Blocks(title="Autonomous SRE Control Center", css=custom_css) as demo:
        
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

        with gr.Row():
            # Left: Architecture & Signals
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["sre-panel"]):
                    gr.Markdown("<div class='metric-label'>🚨 Incident Alerts</div>")
                    alerts_display = gr.HTML("")
                    gr.Markdown("<div class='metric-label'>📡 Node Signals</div>")
                    signals_display = gr.JSON(label=None, show_label=False)
                
                with gr.Column(elem_classes=["sre-panel"]):
                    gr.Markdown("<div class='metric-label'>🏗️ Deployment Topology</div>")
                    world_model_display = gr.JSON(label=None, show_label=False)

            # Right: Diagnostics & Actions
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

        # Events
        reset_btn.click(
            reset_center, inputs=[task_dd],
            outputs=[sla_display, impact_display, test_display, status_bar, stack_display, world_model_display, signals_display, alerts_display]
        )
        
        chaos_btn.click(
            trigger_chaos, outputs=[sla_display, stack_display, alerts_display]
        )
        
        submit_btn.click(
            process_protocol,
            inputs=[action_type, path_box, diff_box, sub_task_box, checkpoint_box, query_box, signal_box, justification_box],
            outputs=[sla_display, impact_display, test_display, status_bar, stack_display, world_model_display, signals_display, alerts_display],
        )

    return demo
