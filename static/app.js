const featureNames = ["Time", "Amount"];
for (let i = 1; i <= 28; i++) {
    featureNames.push(`V${i}`);
}

const basicGrid = document.getElementById("basicGrid");
const featureGrid = document.getElementById("featureGrid");
const form = document.getElementById("predictForm");
const resultDiv = document.getElementById("result");

function createInputField(name) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", name);
    label.textContent = name;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.id = name;
    input.name = name;
    input.value = "0";

    wrapper.appendChild(label);
    wrapper.appendChild(input);

    return wrapper;
}

function createInputs() {
    featureNames.forEach((name) => {
        const field = createInputField(name);

        if (name === "Time" || name === "Amount") {
            basicGrid.appendChild(field);
        } else {
            featureGrid.appendChild(field);
        }
    });
}

function getInputValue(name) {
    const value = document.getElementById(name).value;
    return value === "" ? 0 : parseFloat(value);
}

function getPayload() {
    const payload = {};
    featureNames.forEach((name) => {
        payload[name] = getInputValue(name);
    });
    return payload;
}

function resetResult() {
    resultDiv.className = "result hidden";
    resultDiv.innerHTML = "";
}

function clearForm() {
    featureNames.forEach((name) => {
        document.getElementById(name).value = "0";
    });
    resetResult();
}

function fillLegitSample() {
    clearForm();

    document.getElementById("Time").value = "10000";
    document.getElementById("Amount").value = "25.50";

    for (let i = 1; i <= 28; i++) {
        document.getElementById(`V${i}`).value = "0";
    }

    document.getElementById("V3").value = "0.12";
    document.getElementById("V7").value = "-0.08";
    document.getElementById("V14").value = "0.05";
}

function fillFraudSample() {
    clearForm();

    document.getElementById("Time").value = "50000";
    document.getElementById("Amount").value = "2500";
    document.getElementById("V1").value = "-3.5";
    document.getElementById("V2").value = "2.8";
    document.getElementById("V3").value = "-4.2";
    document.getElementById("V4").value = "3.1";
    document.getElementById("V10").value = "-2.4";
    document.getElementById("V12").value = "-1.7";
    document.getElementById("V14").value = "-3.8";
    document.getElementById("V17").value = "-2.1";
}

function getRiskLevel(probability) {
    if (probability >= 0.75) return "High";
    if (probability >= 0.25) return "Medium";
    return "Low";
}

function renderResult(data) {
    const prob = Number(data.fraud_probability || 0);
    const threshold = Number(data.threshold || 0);
    const risk = getRiskLevel(prob);
    const decision = data.decision || "UNKNOWN";
    const decisionClass = decision === "FRAUD" ? "fraud" : "legit";

    resultDiv.className = `result ${decisionClass}`;
    resultDiv.classList.remove("hidden");

    resultDiv.innerHTML = `
        <h3>Prediction Result</h3>
        <div class="result-grid">
            <div class="result-card">
                <span class="result-label">Decision</span>
                <span class="result-value">${decision}</span>
            </div>
            <div class="result-card">
                <span class="result-label">Fraud Probability</span>
                <span class="result-value">${prob.toFixed(4)}</span>
            </div>
            <div class="result-card">
                <span class="result-label">Threshold</span>
                <span class="result-value">${threshold}</span>
            </div>
            <div class="result-card">
                <span class="result-label">Risk Level</span>
                <span class="result-value">${risk}</span>
            </div>
            <div class="result-card">
                <span class="result-label">False Positive Cost</span>
                <span class="result-value">${data.cost_policy?.cost_fp ?? "N/A"}</span>
            </div>
            <div class="result-card">
                <span class="result-label">False Negative Cost</span>
                <span class="result-value">${data.cost_policy?.cost_fn ?? "N/A"}</span>
            </div>
        </div>
        <p class="result-note">
            The decision is based on the model probability and the configured cost-sensitive threshold.
        </p>
    `;
}

function renderError(message) {
    resultDiv.className = "result fraud";
    resultDiv.classList.remove("hidden");
    resultDiv.innerHTML = `
        <h3>Prediction Error</h3>
        <p class="result-note">${message}</p>
    `;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resetResult();

    const payload = getPayload();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            const errorMessage = data.detail || "Request failed.";
            renderError(errorMessage);
            return;
        }

        renderResult(data);
    } catch (error) {
        renderError(error.message || "Something went wrong while calling the API.");
    }
});

createInputs();