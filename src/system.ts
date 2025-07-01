import os from "os";

export default function getRamBytes(): number {
  return os.totalmem();
  // for testing return 128 GB
  // return 128 * 1024 * 1024 * 1024; // 128 GB in bytes
}
