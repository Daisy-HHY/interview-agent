import type { SecretStorage } from "vscode";

/** 仅使用宿主提供的加密读写，不维护本地明文文件。 */
type Secrets = Pick<SecretStorage, "get" | "store">;

/** 规范端点标识；空配置等同于官方默认端点。 */
function endpoint(baseUrl: string): string {
  return new URL(baseUrl.trim() || "https://api.openai.com/v1").href.replace(/\/+$/, "");
}

/** 保存一个与当前端点绑定的凭证；切换服务必须重新提供密钥。 */
export async function saveApiKey(secrets: Secrets, baseUrl: string, apiKey: string): Promise<void> {
  try {
    if (!apiKey.trim()) { throw new Error("empty"); }
    await secrets.store("interview.modelCredential", JSON.stringify({
      endpoint: endpoint(baseUrl), apiKey: apiKey.trim(),
    }));
  } catch {
    throw new Error("API Key 加密存储失败，请检查系统凭证服务与 Base URL 后重试。");
  }
}

/** 异步读取端点凭证；首次兼容旧配置但不删除原设置，迁移失败不降级。 */
export async function readApiKey(secrets: Secrets, baseUrl: string, legacy: string): Promise<string> {
  try {
    const value = await secrets.get("interview.modelCredential");
    if (value !== undefined) {
      const stored = JSON.parse(value);
      if (!stored || typeof stored.apiKey !== "string" || typeof stored.endpoint !== "string") {
        throw new Error("invalid credential");
      }
      return stored.endpoint === endpoint(baseUrl) ? stored.apiKey : "";
    }
    if (legacy.trim()) {
      await saveApiKey(secrets, baseUrl, legacy);
      return legacy.trim();
    }
    return "";
  } catch {
    throw new Error("API Key 加密存储读取或迁移失败，原配置已保留，请重试。");
  }
}
