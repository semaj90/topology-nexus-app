import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

interface DocEntry {
  path: string;
  description: string;
  tags?: string[];
  profiles?: string[];
  order?: number;
}

interface MetadataFile {
  docs: DocEntry[];
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const repoRoot = path.resolve(__dirname, "..", "..");
const llmsDir = path.join(repoRoot, "docs", "llms");
const metadataPath = path.join(llmsDir, "index.json");

const profileArg = process.argv.find((arg) => arg.startsWith("--profile="));
const profile = profileArg ? profileArg.split("=")[1] : undefined;

function normaliseProfile(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }
  return value.toLowerCase();
}

function filterByProfile(docs: DocEntry[], profileName: string | undefined): DocEntry[] {
  if (!profileName) {
    return docs;
  }
  return docs.filter((doc) => {
    if (!doc.profiles || doc.profiles.length === 0) {
      return true;
    }
    return doc.profiles.map((entry) => entry.toLowerCase()).includes(profileName);
  });
}

function summarise(content: string, limit: number): string {
  if (content.length <= limit) {
    return content.trim();
  }
  const truncated = content.slice(0, limit);
  const lastSpace = truncated.lastIndexOf(" ");
  const safeSlice = lastSpace > 0 ? truncated.slice(0, lastSpace) : truncated;
  return `${safeSlice.trim()}…`;
}

async function loadDocs(): Promise<DocEntry[]> {
  const metadataRaw = await fs.readFile(metadataPath, "utf-8");
  const metadata = JSON.parse(metadataRaw) as MetadataFile;
  if (!metadata.docs || !Array.isArray(metadata.docs)) {
    throw new Error("index.json must expose a 'docs' array");
  }
  const docs = metadata.docs.slice().sort((a, b) => {
    const aOrder = a.order ?? Number.MAX_SAFE_INTEGER;
    const bOrder = b.order ?? Number.MAX_SAFE_INTEGER;
    return aOrder - bOrder;
  });
  return docs;
}

async function readDocContent(doc: DocEntry): Promise<string> {
  const absolutePath = path.join(repoRoot, doc.path);
  const buffer = await fs.readFile(absolutePath);
  if (doc.path.endsWith(".json") || doc.path.endsWith(".jsonl")) {
    return buffer.toString("utf-8");
  }
  return buffer.toString("utf-8");
}

async function buildLlmsFiles(): Promise<void> {
  const profileName = normaliseProfile(profile);
  const docs = filterByProfile(await loadDocs(), profileName);

  const llmsListLines: string[] = [];
  const fullSections: string[] = [];
  const mediumSections: string[] = [];
  const smallSections: string[] = [];

  for (const doc of docs) {
    const tags = doc.tags && doc.tags.length > 0 ? `tags=${doc.tags.join(",")}` : "";
    llmsListLines.push(`${doc.path} | ${doc.description}${tags ? ` | ${tags}` : ""}`);

    const content = await readDocContent(doc);

    const header = [`# ${doc.path}`, doc.description, tags ? `Tags: ${tags}` : ""].filter(Boolean).join("\n");
    fullSections.push(`${header}\n\n${content.trim()}\n`);

    mediumSections.push(
      `${header}\n\n${summarise(content, 1200)}\n`
    );

    smallSections.push(
      `${header}\n\n${summarise(content, 400)}\n`
    );
  }

  const outputs: Record<string, string> = {
    "llms.txt": llmsListLines.join("\n") + "\n",
    "llms-full.txt": fullSections.join("\n\n"),
    "llms-medium.txt": mediumSections.join("\n\n"),
    "llms-small.txt": smallSections.join("\n\n"),
  };

  for (const [fileName, data] of Object.entries(outputs)) {
    const outputPath = path.join(repoRoot, fileName);
    await fs.writeFile(outputPath, data, "utf-8");
  }

  const profileLabel = profileName ? ` for profile "${profileName}"` : "";
  console.log(`Generated llms documentation bundles${profileLabel}.`);
}

buildLlmsFiles().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
