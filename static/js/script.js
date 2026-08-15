
// Render initial static content if passed via standard template render
const rawResultElem = document.getElementById('raw-result');
const rawFeedbackElem = document.getElementById('raw-feedback');

if (rawResultElem && rawResultElem.textContent.trim()) {
    document.getElementById('rendered-result').innerHTML = marked.parse(rawResultElem.textContent);
}
if (rawFeedbackElem && rawFeedbackElem.textContent.trim()) {
    document.getElementById('rendered-feedback').innerHTML = marked.parse(rawFeedbackElem.textContent);
}

// Live SSE Streaming Handler
document.querySelector("form").addEventListener("submit", async (e) => {
    e.preventDefault(); // Intercept default form submit

    const form = e.target;
    const formData = new FormData(form);

    // const outputContainer = document.getElementById("output-container");
    const logBox = document.getElementById("live-logs");
    const renderedResult = document.getElementById("rendered-result");
    const renderedFeedback = document.getElementById("rendered-feedback");
    const terminalLogsList = document.getElementById("terminal-logs-list");
    const systemContextList = document.getElementById("system-context-list");

    const sidebarContainer = document.getElementById("sidebar-container");
    const responseCard = document.getElementById("response-card");

    // Reveal container and reset views
    // outputContainer.style.display = "block";
    renderedResult.innerHTML = "<em>Generating response...</em>";
    renderedFeedback.innerHTML = "<em>Awaiting evaluation...</em>";
    terminalLogsList.innerHTML = "<li>> Executing workflow...</li>";
    systemContextList.innerHTML = "<li>Gathering execution context...</li>";

    if (sidebarContainer) sidebarContainer.style.display = "flex";
    if (responseCard) responseCard.style.display = "block";

    // Sliding Window Configuration
    const MAX_LOGS = 20;
    let logQueue = [];
    logBox.innerHTML = "Initializing workflow...";
    
    try {
        // Initiate POST request to the streaming endpoint
        const response = await fetch('/chat/stream', {
            method: 'POST',
            body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';    
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            // Add the new chunk to whatever was left over in the buffer
            buffer += decoder.decode(value, { stream: true });
            
            // Split by newline
            const lines = buffer.split('\n');
            
            // The very last item in the array is either an empty string (if it ended cleanly)
            // or a partial broken line. We pop it off and keep it in the buffer for the next loop.
            buffer = lines.pop(); 

            for (let line of lines) {
                if (line.trim().startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'log') {
                            // Update rolling live trace window
                            logQueue.push(`> ${data.message}`);
                            if (logQueue.length > MAX_LOGS) {
                                logQueue.shift();
                            }
                            logBox.innerHTML = logQueue.join('<br>');
                            // Auto-scroll to the bottom of the terminal
                            logBox.scrollTop = logBox.scrollHeight;

                        } else if (data.type === 'complete') {
                            // Final trace status update
                            logQueue.push(`> 🎉 Workflow Execution Completed.`);
                            logBox.innerHTML = logQueue.join('<br>');

                            // Render Final Markdown Answer
                            if (data.final_response) {
                                renderedResult.innerHTML = marked.parse(data.final_response);
                            }

                            // Render Critic Feedback
                            if (data.feedback) {
                                renderedFeedback.innerHTML = marked.parse(data.feedback);
                            }

                            // Render Full Action Logs Array if provided in payload
                            if (data.action_logs && data.action_logs.length > 0) {
                                terminalLogsList.innerHTML = data.action_logs
                                    .map(log => `<li>> ${log}</li>`)
                                    .join('');
                            } else {
                                terminalLogsList.innerHTML = "<li>> Execution trace complete.</li>";
                            }

                            // Render System Context History Array if provided in payload
                            if (data.context && data.context.length > 0) {
                                systemContextList.innerHTML = data.context
                                    .map(item => `<li style="margin-bottom: 10px;">${item}</li>`)
                                    .join('');
                            } else {
                                systemContextList.innerHTML = "<li>No worker payloads captured.</li>";
                            }
                        }
                    } catch (parseError) {
                        // If a weird chunk still gets through, log it instead of crashing the whole stream
                        console.warn("Skipped malformed stream chunk:", line);
                    }
                }
            }
        }
    } catch (err) {
        console.error("Stream failed:", err);
        logBox.innerHTML = `> ❌ Connection Error: ${err.message}`;
    }
});

// Toggle Modal Visibility and Fetch latest files
async function toggleFilesModal() {
    const modal = document.getElementById("files-modal");
    if (modal.style.display === "none" || modal.style.display === "") {
        modal.style.display = "block";
        await loadFiles();
    } else {
        modal.style.display = "none";
    }
}

// Fetch files from FastAPI backend
async function loadFiles() {
    const tbody = document.getElementById("files-tbody");
    tbody.innerHTML = "<tr><td colspan='2'>Loading...</td></tr>";

    try {
        const response = await fetch('/api/downloads');
        const data = await response.json();

        tbody.innerHTML = "";
        if (data.files.length === 0) {
            tbody.innerHTML = "<tr><td colspan='2'>No files downloaded yet.</td></tr>";
            return;
        }

        data.files.forEach(file => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                    <td><a href="${file.url}" target="_blank" style="text-decoration: none; color: #0366d6;">${file.name}</a></td>
                    <td><button class="delete-btn" onclick="deleteFile('${encodeURIComponent(file.name)}')">X</button></td>
                `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        tbody.innerHTML = `<tr><td colspan='2'>Error loading files: ${error.message}</td></tr>`;
    }
}

// Delete file via FastAPI backend
async function deleteFile(filename) {
    if (!confirm("Are you sure you want to delete this file?")) return;

    try {
        const response = await fetch(`/api/downloads/${filename}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            await loadFiles(); // Refresh table on success
        } else {
            alert("Failed to delete file.");
        }
    } catch (error) {
        alert("Error: " + error.message);
    }
}

// Client-side Search/Filter logic
function filterFiles() {
    const input = document.getElementById("file-search");
    const filter = input.value.toLowerCase();
    const tbody = document.getElementById("files-tbody");
    const trs = tbody.getElementsByTagName("tr");

    for (let i = 0; i < trs.length; i++) {
        const td = trs[i].getElementsByTagName("td")[0];
        if (td) {
            const txtValue = td.textContent || td.innerText;
            if (txtValue.toLowerCase().indexOf(filter) > -1) {
                trs[i].style.display = "";
            } else {
                trs[i].style.display = "none";
            }
        }
    }
}

