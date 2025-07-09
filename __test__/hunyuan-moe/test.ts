import Log from "../../src/log.ts";
import optimize from "../../src/optimize.ts";
import {
  assertEqual,
  baselineContextLength,
  baselineContextQuantizationSize,
  baselineGpus,
  baselineRamBytes,
} from "../test-utils.ts";
import dump from "./dump.json" with { type: "json" };

export function testHunyuanAllocates() {
  Log.noLog = true;
  const result = optimize({
    gguf: dump as any,
    gpus: baselineGpus(),
    ramBytes: baselineRamBytes(),
    check: true,
    gpuPercentage: 1,
    contextLength: baselineContextLength(),
    contextQuantizationSize: baselineContextQuantizationSize(),
  });
  let totalBytesAllocated = 0;
  for (const device of result.deviceAllocation) {
    totalBytesAllocated += device.bytesAllocated;
  }
  assertEqual(totalBytesAllocated > 0, true, "No bytes were allocated");
}

export function testHunyuanNoGpuOverrun() {
  Log.noLog = true;
  const result = optimize({
    gguf: dump as any,
    gpus: baselineGpus(),
    ramBytes: baselineRamBytes(),
    check: true,
    gpuPercentage: 1,
    contextLength: baselineContextLength(),
    contextQuantizationSize: baselineContextQuantizationSize(),
  });
  for (const device of result.deviceAllocation) {
    assertEqual(
      device.bytesAllocated < device.memoryTotalBytes,
      true,
      `Device ${device.name} has overrun: ${device.bytesAllocated} >= ${device.memoryTotalBytes}`
    );
  }
}
