"""
Premium Startup-Grade UI for CodeOrganismVM — Boardroom Edition.

Implements a sleek, dark-mode dashboard using heavy CSS overrides, 
glassmorphism, and a professional 3-column telemetry layout.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any

import gradio as gr

from models import Action, CodeOrganismActionType
from environment import CodeOrganismEnv, VITALITY_COSTS


def create_gradio_app() -> gr.Blocks:
    env = CodeOrganismEnv()

    # ── Premium UI Design System ──────────────────────────────────────────────

    custom_css = """
    /* Main container and Dark Mode optimization */
    .gradio-container { 
        background: #0f172a !important; 
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    
    /* Glassmorphism panels */
    .glass-panel {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .card-title {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Professional Terminal Look */
    .terminal-box {
        background: #000000 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        color: #10b981 !important;
        padding: 12px !important;
    }

    /* Buttons */
    .primary-btn {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
        border-radius: 12px !important;
    }
    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    
    /* Vitality Animations */
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-4px); }
        40%, 80% { transform: translateX(4px); }
    }
    
    @keyframes pulse-critical {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes thrival-glow {
        0% { box-shadow: 0 0 5px #10b981; }
        50% { box-shadow: 0 0 20px #10b981; }
        100% { box-shadow: 0 0 5px #10b981; }
    }

    .damage-shake { animation: shake 0.4s ease-in-out; }
    .critical-pulse { animation: pulse-critical 2s infinite; }
    .thriving-aura { animation: thrival-glow 3s infinite; }

    /* Tables */
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; color: #94a3b8; font-size: 0.8rem; padding: 8px; border-bottom: 1px solid #334155; }
    td { padding: 8px; font-size: 0.9rem; border-bottom: 1px solid #1e293b; }
    """

    def get_vitality_html(v: float) -> str:
        color = "#10b981"  # Emerald
        animation_class = ""
        label = "SYSTEM STABLE"
        
        if v <= 0:
            color = "#ef4444"
            label = "TERMINATED"
        elif v < 30:
            color = "#ef4444"
            animation_class = "critical-pulse damage-shake"
            label = "CRITICAL FAILURE"
        elif v < 70:
            color = "#f59e0b"
            label = "DEGRADED"
        elif v == 100:
            animation_class = "thriving-aura"
            label = "OPTIMAL THIVAL"

        return f'''
        <div style="padding: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-family: monospace;">
                <span style="color: {color}; font-weight: 800;">{label}</span>
                <span style="color: #94a3b8;">{v:.1f}% VITALITY</span>
            </div>
            <div style="width: 100%; background: #1e293b; border-radius: 99px; height: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05);">
                <div class="{animation_class}" style="width: {v}%; background: {color}; height: 100%; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
        </div>
        '''

    def format_reward_breakdown_html(b: Any) -> str:
        if not b: return "<p style='color: #64748b;'>No data available</p>"
        
        def row(label, val, weight):
            color = "#10b981" if val > 0 else "#ef4444" if val < 0 else "#64748b"
            return f'''
            <tr>
                <td style="color: #94a3b8;">{label}</td>
                <td style="color: {color}; font-weight: 700; text-align: right;">{val:+.2f}</td>
                <td style="color: #475569; font-size: 0.7rem; text-align: right;">{weight}</td>
            </tr>
            '''

        return f'''
        <table style="width: 100%;">
            <thead>
                <tr>
                    <th style="width: 50%;">METRIC</th>
                    <th style="text-align: right;">VALUE</th>
                    <th style="text-align: right;">WT</th>
                </tr>
            </thead>
            <tbody>
                {row("Vitality", b.vitality_delta, "35%")}
                {row("Recovery", b.test_recovery, "30%")}
                {row("Efficiency", b.efficiency_bonus, "15%")}
                {row("Teaming", b.coordination_bonus, "10%")}
                {row("Novelty", b.novelty_bonus, "10%")}
                <tr style="border-top: 2px solid #334155;">
                    <td style="color: #f43f5e; font-weight: 800; padding-top: 12px;">WATCHDOG</td>
                    <td colspan="2" style="color: #f43f5e; font-weight: 800; text-align: right; padding-top: 12px;">{b.watchdog_penalty:+.1f}</td>
                </tr>
            </tbody>
        </table>
        '''

    # ── Business Logic Callbacks ──────────────────────────────────────────────

    def reset_env(task_id):
        obs = env.reset(task_id)
        
        vitality_html = get_vitality_html(obs.vitality_score)
        
        test_text = "### 🧪 Automated Test Suite\n\n| ID | STATUS | DIAGNOSTIC |\n| :--- | :--- | :--- |\n"
        for t in obs.test_results:
            icon = "🟢" if t.status == "PASS" else "🔴" if t.status == "FAIL" else "🟡"
            test_text += f"| {t.name} | {icon} {t.status} | {t.message or 'N/A'} |\n"
            
        tree_text = "### 📁 Biosynthetic Source Tree\n"
        for f in obs.file_tree:
            icon = "🧬" if f.modified_at > 0 else "📄"
            tree_text += f"- {icon} `{f.path}`\n"
            
        status = f"**LIFECYCLE:** {task_id.upper()} | **PHASE:** {env._phase_num} | **EPISODE:** {env._episode_id}"
        
        return (
            vitality_html,
            test_text,
            tree_text,
            obs.stack_trace or "System nominal. No exceptions.",
            status,
            format_reward_breakdown_html(None),
            "0.0000",
            obs.dependency_graph,
            obs.recent_signals,
            [] # Alerts
        )

    def submit_action(
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
            return None, f"### ⚠️ ERROR\n{e}", "", "", "SYSTEM_FAILURE", "", "0.0000", {}, [], ["Runtime Error Exception"]

        obs = result.observation
        b = result.reward_breakdown
        st = env.state()
        
        vitality_html = get_vitality_html(obs.vitality_score)
        
        test_text = "### 🧪 Automated Test Suite\n\n| ID | STATUS | DIAGNOSTIC |\n| :--- | :--- | :--- |\n"
        for t in obs.test_results:
            icon = "🟢" if t.status == "PASS" else "🔴" if t.status == "FAIL" else "🟡"
            test_text += f"| {t.name} | {icon} {t.status} | {t.message or 'N/A'} |\n"
            
        tree_text = "### 📁 Biosynthetic Source Tree\n"
        for f in obs.file_tree:
            icon = "🧬" if f.modified_at > 0 else "📄"
            tree_text += f"- {icon} `{f.path}`\n"
            
        status_info = f"**STEP:** {obs.timestep}/{obs.max_steps} | **SCORE:** {st.cumulative_reward:.4f}"
        
        if result.done:
            term = result.info.get("termination", "timeout")
            status_info = f"### 🏁 EPISODE {term.upper()} | SCORE: {st.cumulative_reward:.4f}"
            if term == "organism_thrival": vitality_html = get_vitality_html(100.0)

        # Build Alert UI
        alert_html = ""
        if obs.alerts:
            for a in obs.alerts:
                alert_html += f"<div style='background: rgba(244, 63, 94, 0.1); border-left: 4px solid #f43f5e; padding: 8px; margin-bottom: 8px; font-size: 0.85rem;'>{a}</div>"

        return (
            vitality_html,
            test_text,
            tree_text,
            obs.stack_trace or "No active exceptions.",
            status_info,
            format_reward_breakdown_html(b),
            f"{result.reward:.4f}",
            obs.dependency_graph,
            obs.recent_signals,
            alert_html
        )

    # ── Dashboard Layout ──────────────────────────────────────────────────────

    with gr.Blocks(
        title="CodeOrganismVM — Strategic Command",
        theme=gr.themes.Default(primary_hue="emerald", secondary_hue="slate"),
        css=custom_css
    ) as demo:
        
        with gr.Row(elem_classes=["glass-panel"], variant="panel"):
            with gr.Column(scale=4):
                gr.Markdown("# 🧬 CodeOrganismVM")
                gr.Markdown("*Real-time Telemetry & Strategic Alignment Interface*")
            with gr.Column(scale=1):
                task_dd = gr.Dropdown(["phase_1", "phase_2", "phase_3"], value="phase_1", label="Curriculum")
                reset_btn = gr.Button("REBIRTH SEQUENCE", variant="secondary", elem_classes=["primary-btn"])

        with gr.Row():
            with gr.Column(scale=3):
                vitality_display = gr.HTML(get_vitality_html(100))
            with gr.Column(scale=1):
                status_bar = gr.Markdown("Waiting for sequence start...", elem_classes=["card-title"])

        with gr.Row():
            # Column 1: System Alerts & Signals
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("<div class='card-title'>📡 Telemetry Alerts</div>")
                alerts_display = gr.HTML("")
                gr.Markdown("<div class='card-title'>📡 Neural Signals</div>")
                signals_display = gr.JSON(label=None, show_label=False)

            # Column 2: Core Diagnostics
            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                with gr.Tabs():
                    with gr.Tab("Tests"):
                        test_display = gr.Markdown("Execute Rebirth to initialize suite.")
                    with gr.Tab("Stack Trace"):
                        stack_display = gr.Markdown("", elem_classes=["terminal-box"])
                    with gr.Tab("World Model"):
                        world_model_display = gr.JSON(label=None, show_label=False)

            # Column 3: State & Reward
            with gr.Column(scale=1, elem_classes=["glass-panel"]):
                gr.Markdown("<div class='card-title'>💰 Reward Signal</div>")
                reward_display = gr.HTML(format_reward_breakdown_html(None))
                score_display = gr.Textbox(label="Last Step Delta", interactive=False)
                gr.Markdown("<div class='card-title'>🌳 State Hierarchy</div>")
                tree_display = gr.Markdown("")

        # Action Console
        with gr.Column(elem_classes=["glass-panel"], variant="panel"):
            gr.Markdown("### 🕹 COMMAND CONSOLE")
            with gr.Row():
                action_type = gr.Dropdown([e.value for e in CodeOrganismActionType], value="patch_file", label="OP_CODE")
                path_box = gr.Textbox(label="TARGET_PATH", placeholder="src/core.py")
                signal_box = gr.Textbox(label="SIGNAL_TYPE", placeholder="INTENT_PATCH")
                checkpoint_box = gr.Textbox(label="CHECKPOINT_ID")
            
            with gr.Row():
                diff_box = gr.Textbox(label="DIFF_PAYLOAD", lines=4, placeholder="OLD|NEW")
                justification_box = gr.Textbox(label="JUSTIFICATION", lines=4)
            
            with gr.Row():
                sub_task_box = gr.Textbox(label="SUBAGENT_TASK")
                query_box = gr.Textbox(label="EXPERT_QUERY")
            
            submit_btn = gr.Button("INJECT ACTION SEQUENCE", variant="primary", elem_classes=["primary-btn"])

        # Event Mappings
        reset_btn.click(
            reset_env,
            inputs=[task_dd],
            outputs=[vitality_display, test_display, tree_display, stack_display, status_bar, reward_display, score_display, world_model_display, signals_display, alerts_display],
        )
        
        submit_btn.click(
            submit_action,
            inputs=[action_type, path_box, diff_box, sub_task_box, checkpoint_box, query_box, signal_box, justification_box],
            outputs=[vitality_display, test_display, tree_display, stack_display, status_bar, reward_display, score_display, world_model_display, signals_display, alerts_display],
        )

    return demo
