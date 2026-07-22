import { fetchBacktestSnapshot } from "./api";

describe("typed REST adapter", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards TanStack Query's AbortSignal to fetch", async () => {
    let receivedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
        receivedSignal = init?.signal ?? undefined;
        return new Promise<Response>((_resolve, reject) => {
          receivedSignal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          );
        });
      }),
    );
    const controller = new AbortController();
    const request = fetchBacktestSnapshot(controller.signal, 900);

    controller.abort();

    await expect(request).rejects.toMatchObject({ name: "AbortError" });
    expect(receivedSignal).toBe(controller.signal);
    expect(receivedSignal?.aborted).toBe(true);
  });

  it("accepts only the typed server projection", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            revision: "rest-0001",
            source: "mock-rest",
            serverAuthorization: "fixture-server-projection",
            rows: [],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      ),
    );

    await expect(
      fetchBacktestSnapshot(new AbortController().signal, 0),
    ).resolves.toEqual(
      expect.objectContaining({ revision: "rest-0001", rows: [] }),
    );
  });
});
