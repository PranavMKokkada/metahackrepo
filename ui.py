"""Gradio UI for CodeOrganismVM — Boardroom Edition.

Provides an immersive interface for humans to observe or play the organism:
- Real-time Vitality Bar (animated Green -> Red)
- R1-R5 Reward Breakdown dashboard
- Code Explorer + Live Stack Trace
- Interactive Action Terminal
"""

from __future__ import annotations

import json
from typing import Dict, List, Any

import gradio as gr

from models import Action, CodeOrganismActionType
from environment import CodeOrganismEnv, VITALITY_COSTS


def create_gradio_app() -> gr.Blocks:
    # We'll use a shared environment instance for the UI player
    # In a production multi-user setting, we'd use sessions
    env = CodeOrganismEnv()

    def get_vitality_html(v: float) -> str:
        color = "#22c55e"  # green
        if v < 30:
            color = "#ef4444"  # red
        elif v < 70:
            color = "#f59e0b"  # yellow
            
        return f'''
        <div style="width: 100%; background-color: #334155; border-radius: 8px; height: 32px; overflow: hidden; border: 1px solid #475569;">
            <div style="width: {v}%; background: linear-gradient(90deg, {color} 0%, #10b981 100%); height: 100%; transition: width 0.5s ease-in-out; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                {v:.1f}% VITALITY
            </div>
        </div>
        '''

    def format_reward_breakdown(b: Any) -> str:
        if not b: return "No data"
        lines = [
            f"| Metric | Weight | Value |",
            f"| :--- | :---: | :---: |",
            f"| R1: Vitality | 35% | {b.vitality_delta:+.2f} |",
            f"| R2: Recovery | 30% | {b.test_recovery:+.2f} |",
            f"| R3: Efficiency | 15% | {b.efficiency_bonus:+.2f} |",
            f"| R4: Coordination| 10% | {b.coordination_bonus:+.2f} |",
            f"| R5: Novelty | 10% | {b.novelty_bonus:+.2f} |",
            f"| --- | --- | --- |",
            f"| **Watchdog** | **Penalty** | **{b.watchdog_penalty:+.2f}** |"
        ]
        return "\n".join(lines)

    def reset_env(task_id):
        obs = env.reset(task_id)
        
        # Initial displays
        vitality_html = get_vitality_html(obs.vitality_score)
        
        test_text = "**Test Results:**\n\n| Test | Status | Message |\n|---|---|---|\n"
        for t in obs.test_results:
            status_icon = "✅" if t.status == "PASS" else "❌" if t.status == "FAIL" else "⚠️"
            test_text += f"| {status_icon} {t.name} | {t.status} | {t.message} |\n"
            
        tree_text = "**File Tree:**\n"
        for f in obs.file_tree:
            mod_icon = "📝 " if f.modified_at > 0 else "📄 "
            tree_text += f"- {mod_icon} {f.path}\n"
            
        status = f"Step {obs.timestep}/{obs.max_steps} | Task: {task_id} | Checkpoints: {len(obs.active_checkpoints)}"
        
        stack_trace = obs.stack_trace or "No exceptions currently in buffer."
        
        return (
            vitality_html,
            test_text,
            tree_text,
            stack_trace,
            status,
            "",
            "0.0",
        )

    def submit_action(
        action_type, path, diff, test_suite, sub_task,
        checkpoint_id, query, signal_type, justification
    ):
        try:
            # Construct action
            action = Action(
                action_type=CodeOrganismActionType(action_type),
                path=path or None,
                diff=diff or None,
                test_suite=test_suite or None,
                task=sub_task or None,
                checkpoint_id=checkpoint_id or None,
                query=query or None,
                signal_type=signal_type or None,
                justification=justification or ""
            )
            result = env.step(action)
        except Exception as e:
            return None, f"**Error:** {e}", "", "", "Error", "", "0.0"

        obs = result.observation
        b = result.reward_breakdown
        
        vitality_html = get_vitality_html(obs.vitality_score)
        
        test_text = "**Test Results:**\n\n| Test | Status | Message |\n|---|---|---|\n"
        for t in obs.test_results:
            status_icon = "✅" if t.status == "PASS" else "❌" if t.status == "FAIL" else "⚠️"
            test_text += f"| {status_icon} {t.name} | {t.status} | {t.message} |\n"
            
        tree_text = "**File Tree:**\n"
        for f in obs.file_tree:
            mod_icon = "📝 " if f.modified_at > 0 else "📄 "
            tree_text += f"- {mod_icon} {f.path}\n"
            
        stack_trace = obs.stack_trace or "No active exceptions."
        if result.info.get("action_result"):
            stack_trace = f"**ACTION RESULT:**\n{json.dumps(result.info['action_result'], indent=2)}\n\n---\n\n" + stack_trace

        st = env.state()
        status = (
            f"Step {obs.timestep}/{obs.max_steps} | "
            f"Checkpoints: {len(obs.active_checkpoints)} | "
            f"Cumulative: {st.cumulative_reward:.4f}"
        )
        
        if result.done:
            term = result.info.get("termination", "timeout")
            status = f"**EPISODE COMPLETE — {term.upper()}** | Score: {st.cumulative_reward:.4f}"
            vitality_html = get_vitality_html(env._vitality)

        reward_text = format_reward_breakdown(b)

        return (
            vitality_html,
            test_text,
            tree_text,
            stack_trace,
            status,
            reward_text,
            f"{result.reward:.4f}",
        )

    # CSS for premium look
    custom_css = """
    .vitality-container { margin-bottom: 20px; }
    .card { background: #1e293b; border-radius: 12px; padding: 15px; border: 1px solid #334155; }
    .terminal-output { font-family: 'Courier New', Courier, monospace; font-size: 0.85em; }
    """

    with gr.Blocks(
        title="CodeOrganismVM — Boardroom Edition",
        theme=gr.themes.Default(primary_hue="emerald", secondary_hue="slate"),
        css=custom_css
    ) as demo:
        gr.Markdown(
            "# 🧬 CodeOrganismVM\n"
            "*A program that refuses to die. The agent must self-heal or perish.*"
        )
        
        with gr.Row(variant="compact"):
            vitality_display = gr.HTML(get_vitality_html(100), elem_classes=["vitality-container"])

        with gr.Row():
            task_dd = gr.Dropdown(
                ["phase_1", "phase_2", "phase_3"],
                value="phase_1",
                label="Environment Phase",
            )
            reset_btn = gr.Button("Rebirth Organism", variant="secondary")

        status_bar = gr.Markdown("Click Rebirth to start a new lifecycle")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Tests & Failures"):
                    test_display = gr.Markdown("No tests run", label="Active Test Suite")
                    stack_display = gr.Markdown("", label="Env Logs / Stack Trace")
                with gr.Tab("File System"):
                    tree_display = gr.Markdown("", label="Organism State Tree")
            
            with gr.Column(scale=1):
                reward_display = gr.Markdown("Reward signals will appear here.", label="R1-R5 Breakdown")
                score_display = gr.Textbox(
                    label="Last Step Reward", value="0.0", interactive=False
                )

        gr.Markdown("### 🛠 Biosynthetic Actions")
        with gr.Row():
            action_type = gr.Dropdown(
                [e.value for e in CodeOrganismActionType],
                value="patch_file",
                label="Action"
            )
            path_box = gr.Textbox(label="Path / Module", placeholder="src/core.py")
            checkpoint_box = gr.Textbox(label="Checkpoint ID", placeholder="cp_5")

        with gr.Row():
            diff_box = gr.Textbox(label="Diff / Content", placeholder="old|new or full code", lines=3)
            sub_task_box = gr.Textbox(label="Subagent Task", placeholder="Identify and fix race condition")

        with gr.Row():
            query_box = gr.Textbox(label="Expert Query", placeholder="How to fix this ImportError?")
            signal_box = gr.Textbox(label="Signal Type", placeholder="repair_complete")
            justification_box = gr.Textbox(label="Justification (required)", placeholder="Fixing corruption in auth module")

        submit_btn = gr.Button("Inject Action", variant="primary")

        # Layout organization
        reset_btn.click(
            reset_env,
            inputs=[task_dd],
            outputs=[vitality_display, test_display, tree_display, stack_display, status_bar, reward_display, score_display],
        )
        
        submit_btn.click(
            submit_action,
            inputs=[
                action_type, path_box, diff_box, gr.State(""), sub_task_box,
                checkpoint_box, query_box, signal_box, justification_box
            ],
            outputs=[vitality_display, test_display, tree_display, stack_display, status_bar, reward_display, score_display],
        )

    return demo
