import child_process from "child_process";

export interface Gpu {
  cudaId: number;
  name: string;
  memoryTotalBytes: number;
}

export function getNvidiaGpus(): Gpu[] {
  const output = child_process
    .execSync(
      "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"
    )
    .toString();
  const lines = output.trim().split("\n");
  const gpus: Gpu[] = [];
  for (const line of lines) {
    const [index, name, memoryTotal] = line.split(",").map((s) => s.trim());
    gpus.push({
      cudaId: parseInt(index, 10),
      name,
      memoryTotalBytes:
        parseInt(memoryTotal.replace(" MiB", ""), 10) * 1024 * 1024, // Convert MiB to bytes
    });
  }
  return gpus;
}
