process.removeAllListeners("warning").on("warning", (err) => {
  if (
    err.name !== "ExperimentalWarning" &&
    !err.message.includes("experimental")
  ) {
    console.warn(err);
  }
});

import { gguf } from "@huggingface/gguf";
import { Command } from "commander";
import { getNvidiaGpus } from "./nvidia.ts";
import optimize from "./optimize.ts";
import getRamBytes from "./system.ts";
import { downloadGguf } from "./gguf.ts";
import Log from "./log.ts";

const program = new Command();
program
  .name("nvidia-gguf-optimizer")
  .description("Optimize GGUF files for NVIDIA GPUs")
  .version("1.0.0");
program
  .option("-g, --gguf-url <url>", "URL of the GGUF file to optimize")
  .option("-c, --context-length <length>", "Context length for optimization")
  .option(
    "-q",
    "--context-quantization-size <size>",
    "Context quantization size"
  )
  .option(
    "--no-check",
    "Skip system resource limits check. Useful for when you're using swap"
  )
  .option("--verbose", "Enable verbose logging")
  .action(async (options) => {
    if (options.verbose) {
      Log.vebose = true;
    }
    let ggufUrl = options.ggufUrl;
    if (!ggufUrl) {
      Log.error("GGUF URL is required. Please provide a valid URL.");
      process.exit(1);
    }
    let contextLength = options.contextLength;
    if (!contextLength) {
      Log.error("Context length is required. Please provide a valid length.");
      process.exit(1);
    }
    let contextQuantizationSize = options.contextQuantizationSize || "16";
    if (
      !contextQuantizationSize ||
      !["4", "8", "16"].includes(contextQuantizationSize)
    ) {
      Log.error(
        "Context quantization size must be one of 4, 8, or 16. Please provide a valid size."
      );
      process.exit(1);
    }
    optimize({
      gguf: await downloadGguf(ggufUrl),
      gpus: getNvidiaGpus(),
      ramBytes: getRamBytes(),
      contextLength: +contextLength,
      contextQuantizationSize: +contextQuantizationSize,
      check: options.check,
    });
  });
await program.parseAsync(process.argv);
