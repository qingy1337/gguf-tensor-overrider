export default class Log {
  public static vebose: boolean = false;

  public static log(logLevel: "info" | "default", message: string) {
    if (logLevel === "info" && Log.vebose) {
      console.log(`[${logLevel.toUpperCase()}] ${message}`);
    } else if (logLevel === "default") {
      console.log(message);
    }
  }

  public static warn(message: string) {
    console.warn(`[WARN] ${message}`);
  }

  public static error(message: string) {
    console.error(`[ERROR] ${message}`);
  }
}
