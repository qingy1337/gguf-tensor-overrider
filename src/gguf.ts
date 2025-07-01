import { gguf } from "@huggingface/gguf";
import type { GGUFParseOutput } from "@huggingface/gguf";
import Log from "./log.ts";

export async function downloadGguf(url: string) {
  // Match the pattern: "basename-00001-of-00003.gguf"
  const parts = url.match(/(.+?)-(\d+)-of-(\d+)\.gguf/);

  if (!parts) {
    return gguf(url);
  }

  const baseUrl = parts[1];
  const totalParts = parseInt(parts[3], 10);

  let firstPart: GGUFParseOutput<{ strict: true }> | null = null;

  for (let i = 1; i <= totalParts; i++) {
    const partUrl = `${baseUrl}-${i.toString().padStart(5, "0")}-of-${totalParts
      .toString()
      .padStart(5, "0")}.gguf`;
    Log.log("info", `Downloading part ${i} of ${totalParts} from ${partUrl}`);
    const part = await gguf(partUrl);

    if (firstPart === null) {
      firstPart = part;
    } else {
      firstPart.tensorInfos.push(...part.tensorInfos);
    }
  }

  if (firstPart === null) {
    throw new Error("Failed to download any parts of the GGUF file.");
  }

  return firstPart;
}
