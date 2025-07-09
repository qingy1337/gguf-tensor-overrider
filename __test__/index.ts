import {
  testEverythingAllocated,
  testNoGpuOverrun,
} from "./qwen3-235-q4ks/test.ts";
import {
  testHunyuanAllocates,
  testHunyuanNoGpuOverrun,
} from "./hunyuan-moe/test.ts";

testEverythingAllocated();
testNoGpuOverrun();
testHunyuanAllocates();
testHunyuanNoGpuOverrun();
