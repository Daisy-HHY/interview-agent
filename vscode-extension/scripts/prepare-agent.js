const fs = require("fs");
const path = require("path");

const extensionRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(extensionRoot, "..");
const source = path.join(repoRoot, "agent");
const target = path.join(extensionRoot, "bundled-agent", "agent");

if (!fs.existsSync(source)) {
  throw new Error(`Agent source not found: ${source}`);
}

fs.rmSync(target, { recursive: true, force: true });
fs.mkdirSync(path.dirname(target), { recursive: true });
fs.cpSync(source, target, {
  recursive: true,
  filter: (file) => {
    const normalized = file.replaceAll(path.sep, "/");
    return !(
      normalized.includes("/tests/") ||
      normalized.endsWith("/__pycache__") ||
      normalized.includes("/__pycache__/") ||
      normalized.endsWith(".pyc") ||
      normalized.includes("/.pytest_cache/")
    );
  },
});

fs.writeFileSync(
  path.join(extensionRoot, "bundled-agent", "requirements-agent.txt"),
  [
    "openai>=2.0",
    "",
  ].join("\n"),
  "utf-8",
);

fs.writeFileSync(
  path.join(extensionRoot, "bundled-agent", "requirements-ocr.txt"),
  [
    "PyMuPDF>=1.24",
    "numpy>=1.26",
    "rapidocr>=3.9,<4",
    "onnxruntime>=1.18",
    "",
  ].join("\n"),
  "utf-8",
);

fs.writeFileSync(
  path.join(extensionRoot, "bundled-agent", "requirements-framework.txt"),
  [
    "langchain>=1.0,<2",
    "langchain-openai>=1.0,<2",
    "langgraph>=1.0,<2",
    "",
  ].join("\n"),
  "utf-8",
);

console.log(`Copied Agent runtime to ${target}`);
