import { describe, expect, it, vi } from "vitest";
import { readApiKey, saveApiKey } from "../src/credentials";

/** 模拟 SecretStorage，全部凭证为合成值。 */
function storage() {
  let value: string | undefined;
  return {
    get: vi.fn(async () => value),
    store: vi.fn(async (_name: string, next: string) => { value = next; }),
  };
}

describe("模型凭证加密存储", () => {
  it("新密钥按当前端点保存，切换服务不复用", async () => {
    const secrets = storage();
    await saveApiKey(secrets, "https://provider.test/v1/", " fake-key ");
    expect(await readApiKey(secrets, "https://provider.test/v1", "")).toBe("fake-key");
    expect(await readApiKey(secrets, "https://other.test/v1", "legacy-key")).toBe("");
  });
  it("空默认地址与官方默认端点一致", async () => {
    const secrets = storage();
    await saveApiKey(secrets, "", "fake-key");
    expect(await readApiKey(secrets, "https://api.openai.com/v1", "")).toBe("fake-key");
  });
  it("兼容旧设置但只向加密存储迁移", async () => {
    const secrets = storage();
    expect(await readApiKey(secrets, "", "legacy-fake")).toBe("legacy-fake");
    expect(secrets.store).toHaveBeenCalledOnce();
    expect(await readApiKey(secrets, "", "different-old-value")).toBe("legacy-fake");
  });
  it("没有密钥不会写入或发起网络请求", async () => {
    const secrets = storage();
    expect(await readApiKey(secrets, "", "")).toBe("");
    expect(secrets.store).not.toHaveBeenCalled();
  });
  it("迁移失败不得退回明文继续调用，错误不含密钥", async () => {
    const secrets = storage();
    secrets.store.mockRejectedValue(new Error("fake-secret-in-error"));
    await expect(readApiKey(secrets, "", "fake-secret-in-error"))
      .rejects.toThrow("加密存储");
    await expect(readApiKey(secrets, "", "fake-secret-in-error"))
      .rejects.not.toThrow("fake-secret-in-error");
  });
  it("读取失败或格式损坏明确报错", async () => {
    const secrets = storage();
    secrets.get.mockResolvedValue("broken");
    await expect(readApiKey(secrets, "", "")).rejects.toThrow("加密存储");
  });
});
