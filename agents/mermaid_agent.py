"""agents/mermaid_agent.py — Mermaid Diagram Generator."""
import os
import re
import uuid
import subprocess
from typing import Any, Dict
from workflow.state import MermaidTaskState
DOWNLOAD_DIR = "static/downloads"

def mermaid_agent_node(state: MermaidTaskState) -> Dict[str, Any]:
    """Generates a PNG from a Mermaid script and returns the image link."""
    script = state.get("script", "")
    logs = ["[MERMAID_AGENT] 🎨 Processing Mermaid script..."]
    print("\n[MERMAID_AGENT] 🎨 Processing Mermaid script...")

    # Strip out markdown formatting if the LLM wraps it in ```mermaid ... ```
    script = re.sub(r"^```(?:mermaid)?\s*", "", script, flags=re.IGNORECASE)
    script = re.sub(r"\s*```$", "", script)
    script = script.strip()

    if not script:
        error_msg = "[MERMAID_AGENT] ❌ Error: No script provided."
        logs.append(error_msg)
        print(error_msg)
        return {"context": ["Mermaid Agent Error: No script provided."], "action_logs": logs}

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Generate unique filenames to prevent overwriting
    file_id = str(uuid.uuid4())[:8]
    mmd_filename = f"diagram_{file_id}.mmd"
    png_filename = f"diagram_{file_id}.png"
    
    mmd_path = os.path.join(DOWNLOAD_DIR, mmd_filename)
    png_path = os.path.join(DOWNLOAD_DIR, png_filename)
    web_path = f"/static/downloads/{png_filename}"

    try:
        # 1. Save the Mermaid script to a local file
        with open(mmd_path, "w", encoding="utf-8") as f:
            f.write(script)
        logs.append(f"[MERMAID_AGENT] 💾 Saved script to {mmd_path}")

        # 2. Execute the mmdc CLI command
        # os.name == 'nt' ensures npm commands run correctly on Windows
        cmd = ["mmdc", "-i", mmd_path, "-o", png_path, "-s", "2"]
        use_shell = (os.name == 'nt') 
        
        logs.append(f"[MERMAID_AGENT] ⚙️ Executing: {' '.join(cmd)}")
        print(f"[MERMAID_AGENT] ⚙️ Executing command...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=use_shell)
        
        # 3. Format the explicit success message for the Supervisor
        # Using ![alt](url) will render the image directly in the UI!
        success_message = f"✅ SUCCESS: Mermaid diagram generated. You MUST provide this exact image link to the user: ![Mermaid Diagram]({web_path})"
        
        logs.append("[MERMAID_AGENT] ✅ Diagram generated successfully.")
        print("[MERMAID_AGENT] ✅ Diagram generated successfully.")
        
        return {
            "context": [success_message],
            "action_logs": logs
        }
        
    except subprocess.CalledProcessError as e:
        err_msg = f"[MERMAID_AGENT] ❌ CLI execution failed: {e.stderr}"
        logs.append(err_msg)
        print(err_msg)
        return {"context": [f"Mermaid Agent Error: CLI execution failed - {e.stderr}"], "action_logs": logs}
    except Exception as e:
        err_msg = f"[MERMAID_AGENT] ❌ Unexpected Error: {str(e)}"
        logs.append(err_msg)
        print(err_msg)
        return {"context": [f"Mermaid Agent Error: {str(e)}"], "action_logs": logs}